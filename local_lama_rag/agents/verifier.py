import re
import time

from langchain_core.output_parsers import StrOutputParser

from agents.base import BaseAgent
from prompts.prompt_library import VERIFY_PROMPT
import config


class VerifierAgent(BaseAgent):
    """
    Attaches citations and a confidence score to a generated answer, and
    flags claims that aren't supported by the retrieved context.

    Two-layer approach, cheapest check first:
    1. Lexical overlap (fast, free, no LLM call) -- per-sentence word
       overlap between the answer and the context gives a coarse but
       useful confidence signal with zero added latency.
    2. LLM fact-check (optional, one extra call) -- only run when the
       lexical check finds enough low-overlap sentences to be worth a
       real look, or when the caller explicitly asks for it. This keeps
       the common "answer is clearly well-grounded" case fast, and only
       pays the verification-call cost when there's a real signal of risk.
    """

    def __init__(self, llm, min_overlap: float = None):
        super().__init__()
        self.min_overlap = min_overlap if min_overlap is not None else config.VERIFY_MIN_OVERLAP
        self.chain = VERIFY_PROMPT | llm | StrOutputParser()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    @staticmethod
    def _overlap_ratio(sentence: str, context: str) -> float:
        s_words = set(re.findall(r"\w+", sentence.lower()))
        c_words = set(re.findall(r"\w+", context.lower()))
        if not s_words:
            return 1.0
        return len(s_words & c_words) / len(s_words)

    def lexical_check(self, answer: str, context: str) -> dict:
        sentences = self._split_sentences(answer)
        if not sentences:
            return {"confidence": 0.0, "low_overlap_sentences": []}

        overlaps = [self._overlap_ratio(s, context) for s in sentences]
        low = [s for s, o in zip(sentences, overlaps) if o < self.min_overlap]
        confidence = round(sum(overlaps) / len(overlaps), 2)
        return {"confidence": confidence, "low_overlap_sentences": low}

    def llm_check(self, answer: str, context: str) -> str:
        start = time.time()
        result = self.chain.invoke({"answer": answer, "context": context})
        self.logger.info(f"LLM verification check: {time.time() - start:.2f}s")
        return result.strip()

    def verify(self, answer: str, context: str, sources: list, force_llm_check: bool = False) -> dict:
        """
        Returns {"confidence": 0-1 float, "citations": [...], "flags": [...],
        "unsupported_claims": str or None}. Cheap by default -- only calls
        the LLM fact-checker when the lexical pass finds real risk signal
        or the caller asks for it explicitly (e.g. for a high-stakes answer).
        """
        if not context.strip():
            # No context to verify against (e.g. web/code/general answers) --
            # skip lexical scoring, it would be meaningless.
            return {"confidence": None, "citations": sources, "flags": [], "unsupported_claims": None}

        lexical = self.lexical_check(answer, context)
        flags = lexical["low_overlap_sentences"]

        unsupported = None
        if force_llm_check or len(flags) >= 2:
            result = self.llm_check(answer, context)
            if result.lower() not in ("none", "none.", ""):
                unsupported = result

        return {
            "confidence": lexical["confidence"],
            "citations": sources,
            "flags": flags,
            "unsupported_claims": unsupported,
        }

    @staticmethod
    def cross_check_vision(vlm_description: str, detection_result: dict) -> str | None:
        """
        Compares a vision-language model's free-text description against a
        real object detector's measured count for the same image. This is
        a genuine agentic cross-check, not cosmetic: one path is a language
        model's guess, the other is an actual measurement, and they can
        disagree. Cheap heuristic (no extra LLM call) -- pulls the first
        integer mentioned in the VLM text and compares it to len(detections).
        Returns a short note, or None if there's nothing to flag (no number
        mentioned in the description, or the counts roughly agree).
        """
        if not detection_result or not detection_result.get("success"):
            return None

        detected_count = len(detection_result.get("detections", []))
        numbers_in_text = re.findall(r"\b(\d+)\b", vlm_description or "")
        if not numbers_in_text:
            return None

        mentioned_count = int(numbers_in_text[0])
        if mentioned_count == detected_count:
            return None  # counts agree -- nothing to flag

        return (
            f"Vision-language description mentions {mentioned_count}, but the "
            f"object detector ({detection_result.get('model', 'detector')}) measured "
            f"{detected_count} object(s). Descriptions are qualitative estimates; "
            f"treat the detector's count as the grounded measurement."
        )
