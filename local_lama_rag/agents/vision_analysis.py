"""
==========================================================
Vision Analysis Agent — real CV inference (YOLO)
==========================================================
Runs actual object detection -- not a vision-language model's
free-text caption -- producing structured, measurable output:
class labels, confidence scores, bounding boxes, pixel areas,
and counts. This is what makes the Verifier Agent's cross-check
against ImageAgent's VLM description meaningful: one path is a
language model's description, the other is a real detector's
measurement, and they can genuinely disagree.

Model defaults to YOLO11n (nano) -- CPU-friendly, ~5MB, loads in
under a second after the first download, good enough for a live
demo without needing a GPU.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

from agents.base import BaseAgent

try:
    from ultralytics import YOLO
    from PIL import Image
    import pandas as pd
    _HAS_VISION_STACK = True
except ImportError:
    _HAS_VISION_STACK = False


class VisionAnalysisAgent(BaseAgent):
    """
    Real object detection via YOLO. Lazily loads the model on first use
    so app startup isn't slowed by a model load that many sessions may
    never trigger.
    """

    def __init__(self, weights: str = "yolo26n.pt", confidence: float = 0.25, max_detections: int = 50):
        super().__init__()
        self._weights = weights
        self._confidence = confidence
        self._max_detections = max_detections
        self._model = None
        if not _HAS_VISION_STACK:
            self.logger.warning(
                "ultralytics/Pillow not installed -- VisionAnalysisAgent disabled. "
                "`pip install ultralytics pillow` to enable real object detection."
            )

    @property
    def available(self) -> bool:
        return _HAS_VISION_STACK

    def _ensure_model(self):
        if self._model is None:
            self.logger.info(f"Loading YOLO model: {self._weights} (first call only)")
            start = time.time()
            self._model = YOLO(self._weights)
            self.logger.info(f"Model loaded in {time.time() - start:.2f}s")
        return self._model

    def analyze_bytes(self, image_bytes: bytes, source_label: str = "attached image") -> dict:
        if not self.available:
            return {
                "success": False,
                "answer": "Object detection isn't available (ultralytics not installed).",
                "sources": [source_label], "route": "vision_analysis", "dataframe": None,
            }

        model = self._ensure_model()
        start = time.time()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = model.predict(image, conf=self._confidence, max_det=self._max_detections, verbose=False)
        elapsed = time.time() - start

        result = results[0]
        names = result.names
        rows = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rows.append({
                "class": names.get(cls_id, str(cls_id)),
                "confidence": round(conf, 3),
                "x1": round(x1, 1), "y1": round(y1, 1),
                "x2": round(x2, 1), "y2": round(y2, 1),
                "area_px": round((x2 - x1) * (y2 - y1), 1),
            })

        self.logger.info(f"YOLO inference on {source_label}: {len(rows)} detection(s) in {elapsed:.2f}s")

        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["class", "confidence", "x1", "y1", "x2", "y2", "area_px"]
        )
        counts = df["class"].value_counts().to_dict() if not df.empty else {}

        summary = [f"Detected **{len(df)}** object(s) in {source_label} using {self._weights}."]
        summary.append(
            "Breakdown: " + ", ".join(f"{v}x {k}" for k, v in counts.items())
            if counts else "No objects detected above the confidence threshold."
        )

        return {
            "success": True,
            "answer": " ".join(summary),
            "sources": [source_label],
            "route": "vision_analysis",
            "dataframe": df,
            "detections": rows,
            "model": self._weights,
            "inference_seconds": round(elapsed, 3),
            "counts": counts,
        }

    def analyze_file(self, filename: str, data_dir) -> dict:
        path = Path(data_dir) / filename
        if not path.exists():
            return {"success": False, "answer": f"File not found: {filename}",
                     "sources": [filename], "route": "vision_analysis", "dataframe": None}
        return self.analyze_bytes(path.read_bytes(), source_label=filename)

    def annotated_image_bytes(self, image_bytes: bytes, show_labels: bool = True, show_conf: bool = True) -> bytes | None:
        """Draws detection boxes on the image for UI display. Returns PNG bytes."""
        if not self.available:
            return None
        model = self._ensure_model()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = model.predict(image, conf=self._confidence, max_det=self._max_detections, verbose=False)
        annotated_bgr = results[0].plot(labels=show_labels, conf=show_conf)
        annotated_rgb = annotated_bgr[:, :, ::-1]
        out = io.BytesIO()
        Image.fromarray(annotated_rgb).save(out, format="PNG")
        return out.getvalue()
