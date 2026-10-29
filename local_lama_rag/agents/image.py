import base64
import mimetypes
import time
from pathlib import Path

from agents.base import BaseAgent
from prompts.prompt_library import IMAGE_DESCRIBE_PROMPT, IMAGE_QA_PROMPT

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


class ImageAgent(BaseAgent):
    """
    Understands figures, charts, diagrams, and photos -- both embedded in
    PDFs (extracted via PyMuPDF) and standalone image files uploaded
    directly (indexed or attached ad-hoc in chat). Goes beyond the one-line
    caption generated at ingest time: this agent sends the actual image
    bytes to a vision model along with the user's specific question, so
    "what does the trend in figure 3 show?" gets a real answer instead of
    a generic caption match.
    """

    MAX_IMAGES_PER_CALL = 4  # bound latency when scanning multiple pages/images

    def __init__(self, vision_llm, data_dir):
        super().__init__()
        self.vision_llm = vision_llm
        self.data_dir = Path(data_dir)
        if not _HAS_FITZ:
            self.logger.warning("PyMuPDF (fitz) not installed -- PDF-embedded image extraction disabled "
                                 "(standalone image files still work). `pip install pymupdf`.")

    @property
    def available(self) -> bool:
        return True  # standalone images always work; PDF extraction needs fitz, gated per-call below

    # ── locating images within an indexed file ──────────────────────────────
    def list_images(self, filename: str) -> list[dict]:
        """Returns [{"page", "index", "xref", "standalone"}] for every image
        available in the file -- a single entry for a standalone image file,
        or one per embedded image for a PDF."""
        path = self.data_dir / filename
        if not path.exists():
            return []

        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return [{"page": 0, "index": 0, "xref": None, "standalone": True}]

        if path.suffix.lower() != ".pdf" or not _HAS_FITZ:
            return []

        found = []
        doc = fitz.open(str(path))
        try:
            for page_num in range(len(doc)):
                for img_index, img in enumerate(doc[page_num].get_images(full=True)):
                    found.append({"page": page_num, "index": img_index, "xref": img[0], "standalone": False})
        finally:
            doc.close()
        return found

    def _extract_image_b64(self, filename: str, image_ref: dict) -> tuple[str, str]:
        """Returns (base64_data, mime_type)."""
        path = self.data_dir / filename
        if image_ref.get("standalone"):
            mime = mimetypes.guess_type(str(path))[0] or "image/png"
            return base64.b64encode(path.read_bytes()).decode(), mime

        doc = fitz.open(str(path))
        try:
            pix = fitz.Pixmap(doc, image_ref["xref"])
            if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            return base64.b64encode(pix.tobytes("png")).decode(), "image/png"
        finally:
            doc.close()

    def _vision_call(self, b64_image: str, prompt_text: str, mime: str = "image/png") -> str:
        msg = self.vision_llm.invoke([{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_image}"}},
            ],
        }])
        return msg.content

    def describe(self, filename: str, page: int = None) -> dict:
        images = self.list_images(filename)
        if page is not None:
            images = [i for i in images if i["page"] == page]
        images = images[: self.MAX_IMAGES_PER_CALL]

        if not images:
            return {"answer": f"No images found in {filename}" + (f" on page {page}" if page is not None else "") + ".",
                     "sources": [filename], "route": "image"}

        start = time.time()
        descriptions = []
        for img in images:
            b64, mime = self._extract_image_b64(filename, img)
            desc = self._vision_call(b64, IMAGE_DESCRIBE_PROMPT, mime)
            label = filename if img.get("standalone") else f"Page {img['page'] + 1}, image {img['index'] + 1}"
            descriptions.append(f"**{label}:** {desc}")
        self.logger.info(f"Described {len(images)} image(s) from {filename}: {time.time() - start:.2f}s")

        return {"answer": "\n\n".join(descriptions), "sources": [filename], "route": "image"}

    def answer_question(self, filename: str, question: str, page: int = None) -> dict:
        images = self.list_images(filename)
        if page is not None:
            images = [i for i in images if i["page"] == page]
        images = images[: self.MAX_IMAGES_PER_CALL]

        if not images:
            return {"answer": f"No images found in {filename} to answer that from.",
                     "sources": [filename], "route": "image"}

        prompt_text = IMAGE_QA_PROMPT.format(question=question)
        start = time.time()
        answers = []
        for img in images:
            b64, mime = self._extract_image_b64(filename, img)
            ans = self._vision_call(b64, prompt_text, mime)
            label = filename if img.get("standalone") else f"Page {img['page'] + 1}, image {img['index'] + 1}"
            answers.append(f"**{label}:** {ans}")
        self.logger.info(f"Answered question over {len(images)} image(s) from {filename}: {time.time() - start:.2f}s")

        return {"answer": "\n\n".join(answers), "sources": [filename], "route": "image"}

    # ── ad-hoc images attached directly in chat, never written to data_dir ──
    def describe_bytes(self, image_bytes: bytes, mime: str = "image/png") -> dict:
        start = time.time()
        b64 = base64.b64encode(image_bytes).decode()
        answer = self._vision_call(b64, IMAGE_DESCRIBE_PROMPT, mime)
        self.logger.info(f"Described attached image: {time.time() - start:.2f}s")
        return {"answer": answer, "sources": ["attached image"], "route": "image"}

    def answer_question_bytes(self, image_bytes: bytes, question: str, mime: str = "image/png") -> dict:
        start = time.time()
        b64 = base64.b64encode(image_bytes).decode()
        prompt_text = IMAGE_QA_PROMPT.format(question=question)
        answer = self._vision_call(b64, prompt_text, mime)
        self.logger.info(f"Answered question over attached image: {time.time() - start:.2f}s")
        return {"answer": answer, "sources": ["attached image"], "route": "image"}
