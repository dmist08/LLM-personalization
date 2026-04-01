"""
src/pipeline/rag_baseline.py — BM25 RAG Baseline + No-Personalization Baseline.
=================================================================================
Prompt 6 implementation.

Generates two baselines simultaneously:
  Baseline 1 (No Personalization): Generic prompt, no author context
  Baseline 2 (RAG BM25): Per-author BM25 retrieval with k=2 examples

Uses BASE LLaMA-3.1-8B-Instruct — NO fine-tuning.

Required install:
  pip install rank-bm25

RUN:
  conda activate dl
  python -m src.pipeline.rag_baseline

OUTPUT:
  outputs/baselines/rag_and_base_outputs.jsonl
  logs/rag_baseline_YYYYMMDD_HHMMSS.log

CHECK:
  - Every record has both base_output and rag_output as non-empty strings
  - No output looks like the raw prompt (model echoed prompt = bug)
  - Headline lengths 5–20 words, not full articles
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_config
from src.utils import (
    setup_logging, set_seed, load_jsonl, save_jsonl,
    load_json, get_device, format_article_for_prompt, estimate_runtime,
)

cfg = get_config()
log = setup_logging("rag_baseline", cfg.paths.logs_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# BM25 Per-Author Index
# ═══════════════════════════════════════════════════════════════════════════════

class AuthorBM25Index:
    """BM25 index for a single author's training articles. Never cross-author."""

    def __init__(self, author_id: str, train_articles: list[dict]):
        from rank_bm25 import BM25Okapi

        self.author_id = author_id
        self.articles = train_articles

        # Tokenize: lowercase, split on whitespace
        self._corpus = [
            a.get("article_text", "").lower().split()
            for a in train_articles
        ]

        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
        else:
            self._bm25 = None

    def retrieve(self, query_article: str, k: int = 2) -> list[dict]:
        """Return top-k articles by BM25 similarity."""
        if not self._bm25 or not self.articles:
            return []

        query_tokens = query_article.lower().split()
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        k = min(k, len(self.articles))
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

        return [self.articles[i] for i in top_indices]


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Builders
# ═══════════════════════════════════════════════════════════════════════════════

class RAGPromptBuilder:
    """Builds prompts for both baseline and RAG inference."""

    def __init__(self, max_article_words: int = 150):
        self.max_article_words = max_article_words

    def build_base_prompt(self, new_article: str) -> str:
        """Baseline 1 — no personalization."""
        article = format_article_for_prompt(new_article, self.max_article_words)
        return (
            f"Write a concise news headline for the following article:\n\n"
            f"{article}\n\n"
            f"Headline:"
        )

    def build_rag_prompt(
        self, new_article: str, retrieved_examples: list[dict]
    ) -> str:
        """Baseline 2 — RAG with retrieved examples."""
        if not retrieved_examples:
            return self.build_base_prompt(new_article)

        parts = ["Here are past headlines written by this journalist:\n"]

        for ex in retrieved_examples:
            ex_article = format_article_for_prompt(
                ex.get("article_text", ""), self.max_article_words
            )
            ex_headline = ex.get("headline", "")
            parts.append(f"Article: {ex_article}")
            parts.append(f"Headline: {ex_headline}\n")

        article = format_article_for_prompt(new_article, self.max_article_words)
        parts.append(f"Now write a headline for the following article:")
        parts.append(f"Article: {article}\n")
        parts.append(f"Headline:")

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# RAG Baseline Runner
# ═══════════════════════════════════════════════════════════════════════════════

class RAGBaseline:
    """Runs both Baseline 1 (no context) and Baseline 2 (BM25 RAG)."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        log.info(f"Loading model: {model_name}")
        device = get_device()

        # Load in 8-bit for memory efficiency (~10GB for 8B model)
        if device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()
        self.device = device
        self._indices: dict[str, AuthorBM25Index] = {}
        self.prompt_builder = RAGPromptBuilder(
            max_article_words=cfg.rag.max_article_words_in_context
        )

        if device == "cuda":
            import torch
            allocated = torch.cuda.memory_allocated() / 1e9
            log.info(f"GPU memory after model load: {allocated:.1f} GB")

    def _load_author_indices(self, train_dir: Path) -> dict[str, AuthorBM25Index]:
        """Build BM25 index for each author from per-author train.jsonl files."""
        indices = {}
        author_dirs = [d for d in train_dir.iterdir() if d.is_dir()]

        for author_dir in author_dirs:
            train_file = author_dir / "train.jsonl"
            if not train_file.exists():
                continue

            articles = load_jsonl(train_file)
            if not articles:
                continue

            author_id = author_dir.name
            indices[author_id] = AuthorBM25Index(author_id, articles)

        log.info(f"Built BM25 indices for {len(indices)} authors")
        return indices

    def generate_headline(self, prompt: str, max_new_tokens: int = 30) -> str:
        """Generate one headline. Greedy decoding, no sampling."""
        import torch

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        input_ids = inputs["input_ids"]
        if self.device == "cuda":
            input_ids = input_ids.to("cuda")

        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            else:
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        # Decode only generated tokens
        generated_ids = outputs[0][prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Clean output
        text = text.strip()
        text = re.sub(r'^["\']|["\']$', "", text)  # strip quotes
        text = re.sub(r"^Headline:\s*", "", text, flags=re.IGNORECASE)
        text = text.split("\n")[0].strip()  # take first line only

        return text

    def run_evaluation(
        self,
        test_dir: Path,
        train_dir: Path,
        output_path: Path,
        author_ids: Optional[list[str]] = None,
    ) -> None:
        """
        Run both baselines on all test authors.
        Saves results incrementally to output JSONL.
        """
        # Load BM25 indices
        self._indices = self._load_author_indices(train_dir)

        # Load author metadata for class info
        metadata_path = test_dir.parent / "author_metadata.json"
        metadata = load_json(metadata_path) if metadata_path.exists() else {}

        # Determine which authors to evaluate
        if author_ids:
            eval_authors = author_ids
        else:
            eval_authors = sorted(
                d.name for d in test_dir.iterdir()
                if d.is_dir() and (d / "test.jsonl").exists()
            )

        # Resume: load already-processed authors
        done_authors: set[str] = set()
        if output_path.exists():
            existing = load_jsonl(output_path)
            done_authors = {r["author_id"] for r in existing}
            log.info(f"Resuming: {len(done_authors)} authors already processed")

        remaining = [a for a in eval_authors if a not in done_authors]
        log.info(f"Authors to evaluate: {len(remaining)} (of {len(eval_authors)} total)")

        total_articles = 0
        total_start = time.time()

        with open(output_path, "a", encoding="utf-8") as out_f:
            for author_idx, author_id in enumerate(remaining, 1):
                test_file = test_dir / author_id / "test.jsonl"
                if not test_file.exists():
                    continue

                test_articles = load_jsonl(test_file)
                if not test_articles:
                    continue

                author_meta = metadata.get(author_id, {})
                author_name = author_meta.get("name", author_id)
                author_class = author_meta.get("class", "unknown")
                source = author_meta.get("source", "unknown")

                bm25_index = self._indices.get(author_id)

                author_start = time.time()

                for art in test_articles:
                    article_text = art.get("article_text", "")
                    ground_truth = art.get("headline", "")

                    # Baseline 1: No personalization
                    base_prompt = self.prompt_builder.build_base_prompt(article_text)
                    base_output = self.generate_headline(
                        base_prompt, max_new_tokens=cfg.rag.max_new_tokens
                    )

                    # Baseline 2: RAG (BM25)
                    retrieved = []
                    if bm25_index:
                        retrieved = bm25_index.retrieve(
                            article_text, k=cfg.rag.k_retrieved
                        )

                    rag_prompt = self.prompt_builder.build_rag_prompt(
                        article_text, retrieved
                    )
                    rag_output = self.generate_headline(
                        rag_prompt, max_new_tokens=cfg.rag.max_new_tokens
                    )

                    record = {
                        "author_id": author_id,
                        "author_name": author_name,
                        "author_class": author_class,
                        "source": source,
                        "article_text": article_text,
                        "ground_truth": ground_truth,
                        "base_output": base_output,
                        "rag_output": rag_output,
                        "num_retrieved": len(retrieved),
                    }

                    import json
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_articles += 1

                out_f.flush()
                author_elapsed = time.time() - author_start
                overall_elapsed = time.time() - total_start
                secs_per_article = overall_elapsed / max(total_articles, 1)
                remaining_articles = sum(
                    len(load_jsonl(test_dir / a / "test.jsonl"))
                    for a in remaining[author_idx:]
                    if (test_dir / a / "test.jsonl").exists()
                )
                eta = estimate_runtime(remaining_articles, secs_per_article)

                log.info(
                    f"[{author_idx}/{len(remaining)} authors] {author_name}: "
                    f"{len(test_articles)} test articles done in {author_elapsed:.0f}s | "
                    f"{eta} remaining"
                )

                # Sanity check first author's outputs
                if author_idx == 1:
                    log.info(f"  Sample base_output: '{base_output[:80]}'")
                    log.info(f"  Sample rag_output:  '{rag_output[:80]}'")
                    if len(base_output.split()) > 30:
                        log.warning("  ⚠ base_output looks too long — check prompt stripping")
                    if not base_output:
                        log.error("  ✗ base_output is EMPTY — model may be echoing prompt")

        log.info(f"\nTotal: {total_articles:,} articles, {len(remaining)} authors")
        log.info(f"Output: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAG Baseline Runner")
    parser.add_argument(
        "--model-path", default=cfg.model.base_model,
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--test-dir", default=str(cfg.paths.indian_processed_dir),
        help="Directory with per-author test.jsonl files"
    )
    parser.add_argument(
        "--train-dir", default=str(cfg.paths.indian_processed_dir),
        help="Directory with per-author train.jsonl files"
    )
    parser.add_argument(
        "--output-path",
        default=str(cfg.paths.outputs_dir / "baselines" / "rag_and_base_outputs.jsonl"),
        help="Output JSONL path"
    )
    parser.add_argument(
        "--authors", default=None,
        help="Comma-separated author IDs for partial runs"
    )
    args = parser.parse_args()

    set_seed(cfg.training.seed)

    log.info("=" * 60)
    log.info("RAG BASELINE EVALUATION")
    log.info("=" * 60)
    log.info(f"Model: {args.model_path}")
    log.info(f"Test dir: {args.test_dir}")
    log.info(f"Train dir: {args.train_dir}")
    log.info(f"Output: {args.output_path}")

    author_ids = args.authors.split(",") if args.authors else None

    runner = RAGBaseline(args.model_path)
    runner.run_evaluation(
        test_dir=Path(args.test_dir),
        train_dir=Path(args.train_dir),
        output_path=Path(args.output_path),
        author_ids=author_ids,
    )

    log.info("\n✓ RAG Baseline evaluation complete")
    log.info("Run 'python -m src.pipeline.evaluate' to compute ROUGE-L and METEOR scores")


if __name__ == "__main__":
    main()
