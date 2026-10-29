import re
import time
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser

from agents.base import BaseAgent
from agents.retrieval import RetrievalAgent
from orchestration.model_router import RoutedChain, TaskModelRouter
from prompts.prompt_library import EXTRACT_PROMPT
import config


class TableAgent(BaseAgent):
    """Extracts structured data into tables, explains table contents, and
    can export the result to CSV."""

    def __init__(self, fast_llm, primary_llm, model_router: TaskModelRouter, retrieval_agent: RetrievalAgent):
        super().__init__()
        chain_fast = EXTRACT_PROMPT | fast_llm | StrOutputParser()
        chain_primary = EXTRACT_PROMPT | primary_llm | StrOutputParser()
        self.chain = RoutedChain(model_router, "table", {"fast": chain_fast, "primary": chain_primary})
        self.retrieval_agent = retrieval_agent

    def extract(self, question: str, target_files: list) -> dict:
        if target_files:
            max_per_file = config.SUMMARY_MAX_CHUNKS_PER_FILE
            parts = []
            for f in target_files:
                chunks = self.retrieval_agent.fetch_file_chunks(f, max_per_file)
                if chunks:
                    parts.append(f"=== {f} ===\n" + "\n\n".join(c.page_content for c in chunks))
            context = "\n\n".join(parts)
        else:
            docs, _ = self.retrieval_agent.retrieve(question)
            context = "\n\n".join(d.page_content for d in docs)
            target_files = sorted({d.metadata.get("source_file", "unknown") for d in docs})

        start = time.time()
        table_md = self.chain.invoke({"context": context, "question": question})
        self.logger.info(f"Extraction generation: {time.time() - start:.2f}s")

        df = self.markdown_table_to_dataframe(table_md)
        if df is not None:
            return {
                "answer": f"Extracted {len(df)} row(s) from {', '.join(target_files)}.",
                "sources": target_files, "route": "table", "dataframe": df, "raw_table_md": table_md,
            }
        return {
            "answer": table_md, "sources": target_files, "route": "table", "dataframe": None,
            "raw_table_md": table_md,
        }

    def export_csv(self, df, out_path: str) -> str:
        """Exports a previously extracted DataFrame to CSV. Returns the path."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        self.logger.info(f"Exported table to {out_path}")
        return str(out_path)

    @staticmethod
    def markdown_table_to_dataframe(md_text: str):
        try:
            import pandas as pd

            lines = [ln for ln in md_text.strip().splitlines() if ln.strip().startswith("|")]
            if len(lines) < 2:
                return None

            def split_row(line):
                return [c.strip() for c in line.strip().strip("|").split("|")]

            header = split_row(lines[0])
            data_lines = [ln for ln in lines[1:] if not re.match(r"^\|?[\s:\-|]+\|?$", ln)]
            rows = [split_row(ln) for ln in data_lines]
            rows = [r for r in rows if len(r) == len(header)]
            if not rows:
                return None
            return pd.DataFrame(rows, columns=header)
        except Exception:
            return None
