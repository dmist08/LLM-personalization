"""
deploy.py — Modal LLM endpoint for Cold-Start StyleVector.

Serves all 5 methods LIVE on a single A10G (24GB VRAM):
  1. no_personalization  — plain LLaMA generate
  2. rag_bm25            — generate with author/publication context
  3. stylevector         — activation steering at layer 15
  4. cold_start_sv       — activation steering with interpolated vector
  5. lora_finetuned      — PEFT LoRA adapter generate

Deploy:   modal deploy deploy.py
Dev:      modal serve deploy.py
Test:     modal run deploy.py
Logs:     modal app logs stylevector-llm
"""

import json
import math
import re
from pathlib import Path

import modal

# ── App + Image ───────────────────────────────────────────────────────────────
app = modal.App("stylevector-llm")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_JSONL = PROJECT_ROOT / "data" / "splits" / "indian_train.jsonl"
REMOTE_RAG_TRAIN = "/app/indian_train.jsonl"

llm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "accelerate==0.34.2",
        "peft==0.13.2",
        "numpy>=1.24",
        "fastapi>=0.115",
        "uvicorn>=0.30",
        "sentencepiece>=0.2",
        "protobuf>=4.25",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_file(str(TRAIN_JSONL), remote_path=REMOTE_RAG_TRAIN)
)

# ── Volumes ───────────────────────────────────────────────────────────────────
# Model weights cache — persists across deploys
hf_cache_vol = modal.Volume.from_name("hf-cache-llama", create_if_missing=True)

# Style vectors — uploaded separately via upload_vectors.py
vectors_vol = modal.Volume.from_name("stylevector-data", create_if_missing=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_REPO = "dmist36/llama31-stylevector-lora-indian"  # LoRA adapter on HF Hub
BEST_LAYER = 15
BEST_ALPHA = 0.6       # Steering strength for StyleVector
CS_ALPHA = 0.6         # Cold-start interpolation alpha (from fit)
VECTORS_DIR = "/vectors"
MINUTES = 60
RAG_TOP_K = 2
RAG_MAX_ARTICLE_WORDS = 150

# Prompt — MUST match agnostic_gen.py exactly
AGNOSTIC_PROMPT = (
    "Write ONLY a single neutral, factual news headline for the following article. "
    "Output ONLY the headline text, nothing else. No explanation, no quotes, no prefix.\n\n"
    "{article}\n\nHeadline:"
)


# ── LLM Inference Class ──────────────────────────────────────────────────────
@app.cls(
    image=llm_image,
    gpu="A10G",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        VECTORS_DIR: vectors_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=5 * MINUTES,
    timeout=10 * MINUTES,
    memory=32768,
)
class StyleVectorLLM:
    """Serves all headline generation methods from a single GPU container."""

    @modal.enter()
    def load_model(self):
        """Load base model, LoRA adapter, and style vectors at container start."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import numpy as np

        print("[STARTUP] Loading base model...")
        self.device = torch.device("cuda")

        # Base model — float16 on A10G (24GB)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.base_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"[STARTUP] Base model loaded: {BASE_MODEL}")

        # LoRA adapter
        try:
            self.lora_model = PeftModel.from_pretrained(
                self.base_model, LORA_REPO,
                torch_dtype=torch.float16,
            )
            self.lora_model.eval()
            print(f"[STARTUP] LoRA adapter loaded: {LORA_REPO}")
        except Exception as e:
            print(f"[STARTUP WARN] LoRA adapter failed to load: {e}")
            self.lora_model = None

        # Style vectors — from Modal Volume
        self.style_vectors = {}      # {author_id: np.array(4096,)}
        self.cs_vectors = {}         # {author_id: np.array(4096,)} cold-start
        self.author_metadata = {}

        vec_base = Path(VECTORS_DIR)

        # Load SV vectors for layer 15 (all 42 Indian authors)
        sv_dir = vec_base / "indian" / f"layer_{BEST_LAYER}"
        if sv_dir.exists():
            for f in sv_dir.glob("*.npy"):
                self.style_vectors[f.stem] = np.load(f)
            print(f"[STARTUP] Loaded {len(self.style_vectors)} style vectors (layer {BEST_LAYER})")
        else:
            print(f"[STARTUP WARN] Style vectors not found: {sv_dir}")

        # Load CS vectors (alpha_0.6 — sparse/mid authors only)
        cs_dir = vec_base / "cold_start" / f"alpha_{CS_ALPHA}"
        if cs_dir.exists():
            for f in cs_dir.glob("*.npy"):
                self.cs_vectors[f.stem] = np.load(f)
            print(f"[STARTUP] Loaded {len(self.cs_vectors)} cold-start vectors")
        else:
            print(f"[STARTUP WARN] Cold-start vectors not found: {cs_dir}")

        # Load author metadata
        meta_path = vec_base / "author_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.author_metadata = json.load(f)
            print(f"[STARTUP] Loaded metadata for {len(self.author_metadata)} authors")

        self.rag_indices = self._load_rag_indices(REMOTE_RAG_TRAIN)

        # Commit volume cache after HF downloads
        hf_cache_vol.commit()
        print("[STARTUP] Ready to serve requests!")

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize for BM25 retrieval."""
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _truncate_words(self, text: str, max_words: int = RAG_MAX_ARTICLE_WORDS) -> str:
        words = re.sub(r"\s+", " ", text or "").strip().split()
        return " ".join(words[:max_words])

    def _load_rag_indices(self, train_path: str) -> dict:
        """Build per-author BM25 indexes from Indian training records."""
        by_author = {}
        path = Path(train_path)
        if not path.exists():
            print(f"[STARTUP WARN] RAG train file not found: {train_path}")
            return {}

        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                author_id = str(rec.get("author_id", "")).strip()
                article = rec.get("article_body") or rec.get("article_text") or ""
                headline = rec.get("headline") or ""
                if author_id and article and headline:
                    by_author.setdefault(author_id, []).append({
                        "article": article,
                        "headline": headline,
                    })

        indices = {}
        for author_id, records in by_author.items():
            tokenized = [self._tokenize(r["article"]) for r in records]
            doc_freq = {}
            for tokens in tokenized:
                for tok in set(tokens):
                    doc_freq[tok] = doc_freq.get(tok, 0) + 1

            n_docs = len(records)
            avgdl = sum(len(toks) for toks in tokenized) / max(n_docs, 1)
            idf = {
                tok: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                for tok, df in doc_freq.items()
            }
            indices[author_id] = {
                "records": records,
                "tokenized": tokenized,
                "idf": idf,
                "avgdl": avgdl,
            }

        print(f"[STARTUP] Built RAG BM25 indexes for {len(indices)} authors")
        return indices

    def _bm25_retrieve(self, author_id: str, query: str, k: int = RAG_TOP_K) -> list[dict]:
        """Retrieve top-k same-author training examples by BM25 score."""
        index = self.rag_indices.get(author_id)
        if not index:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        k1 = 1.5
        b = 0.75
        q_terms = set(query_terms)
        scores = []
        avgdl = index["avgdl"] or 1.0

        for doc_idx, tokens in enumerate(index["tokenized"]):
            if not tokens:
                scores.append((0.0, doc_idx))
                continue
            freqs = {}
            for tok in tokens:
                if tok in q_terms:
                    freqs[tok] = freqs.get(tok, 0) + 1

            dl = len(tokens)
            score = 0.0
            for tok, tf in freqs.items():
                idf = index["idf"].get(tok, 0.0)
                denom = tf + k1 * (1 - b + b * dl / avgdl)
                score += idf * (tf * (k1 + 1)) / denom
            scores.append((score, doc_idx))

        top = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
        return [index["records"][idx] for score, idx in top if score > 0]

    def _build_rag_prompt(self, article: str, author_id: str) -> str:
        """Build the same-author BM25 RAG prompt used by the offline baseline."""
        examples = self._bm25_retrieve(author_id, article, k=RAG_TOP_K)
        if not examples:
            article_short = self._truncate_words(article)
            return (
                "Write a concise news headline for the following article:\n\n"
                f"{article_short}\n\nHeadline:"
            )

        parts = ["Here are past headlines written by this journalist:\n"]
        for ex in examples:
            parts.append(f"Article: {self._truncate_words(ex['article'])}")
            parts.append(f"Headline: {ex['headline']}\n")

        parts.append("Now write a headline for the following article:")
        parts.append(f"Article: {self._truncate_words(article)}\n")
        parts.append("Headline:")
        return "\n".join(parts)

    def _generate(self, model, prompt: str, max_tokens: int = 60) -> str:
        """Generate text from a model with the given prompt."""
        import torch

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=700
        ).to(self.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Clean trailing garbage
        for stop in ["\n", " Category:", " Source", " #", "  "]:
            idx = text.find(stop)
            if idx > 5:
                text = text[:idx]
        return text.strip().strip('"\'')

    def _generate_base(self, prompt: str, max_tokens: int = 60) -> str:
        """Generate from the base model with PEFT adapters explicitly disabled."""
        if self.lora_model is not None:
            with self.lora_model.disable_adapter():
                return self._generate(self.lora_model, prompt, max_tokens=max_tokens)
        return self._generate(self.base_model, prompt, max_tokens=max_tokens)

    def _target_layers(self):
        """Return transformer layers for the loaded model, accounting for PEFT wrapping."""
        model = self.lora_model if self.lora_model is not None else self.base_model
        if self.lora_model is not None:
            return model.base_model.model.model.layers
        return model.model.layers

    def _generate_with_steering(self, article: str, author_id: str, use_cold_start: bool = False) -> str:
        """Generate with activation steering at layer BEST_LAYER."""
        import torch
        import numpy as np

        # Pick the right vector
        if use_cold_start:
            meta = self.author_metadata.get(author_id, {})
            author_class = meta.get("class", "unknown")

            if author_class in ("sparse", "mid") and author_id in self.cs_vectors:
                vec = self.cs_vectors[author_id]
            elif author_id in self.style_vectors:
                vec = self.style_vectors[author_id]
            else:
                # Fallback — no vector available
                prompt = AGNOSTIC_PROMPT.format(article=article[:2000])
                return self._generate_base(prompt)
        else:
            if author_id not in self.style_vectors:
                prompt = AGNOSTIC_PROMPT.format(article=article[:2000])
                return self._generate_base(prompt)
            vec = self.style_vectors[author_id]

        # Convert to tensor
        style_tensor = torch.tensor(vec, dtype=torch.float16, device=self.device)

        # Register steering hook
        hook_handle = None

        def steering_hook(module, input, output):
            # output is a tuple: (hidden_states, ...)
            modified = list(output)
            # Add style vector to all generated token positions
            modified[0] = modified[0] + BEST_ALPHA * style_tensor
            return tuple(modified)

        # Hook into the correct layer
        layer = self._target_layers()[BEST_LAYER]
        hook_handle = layer.register_forward_hook(steering_hook)

        try:
            prompt = AGNOSTIC_PROMPT.format(article=article[:2000])
            if self.lora_model is not None:
                with self.lora_model.disable_adapter():
                    result = self._generate(self.lora_model, prompt)
            else:
                result = self._generate(self.base_model, prompt)
        finally:
            if hook_handle:
                hook_handle.remove()

        return result

    @modal.method()
    def generate(self, article: str, method: str, author_id: str = "",
                 publication: str = "") -> dict:
        """Generate a headline using the specified method."""
        import time
        t0 = time.time()

        try:
            if method == "no_personalization":
                prompt = AGNOSTIC_PROMPT.format(article=article[:2000])
                headline = self._generate_base(prompt)

            elif method == "rag_bm25":
                prompt = self._build_rag_prompt(article, author_id)
                headline = self._generate_base(prompt)

            elif method == "stylevector":
                headline = self._generate_with_steering(article, author_id, use_cold_start=False)

            elif method == "cold_start_sv":
                headline = self._generate_with_steering(article, author_id, use_cold_start=True)

            elif method == "lora_finetuned":
                if self.lora_model is None:
                    headline = "[LoRA model not loaded]"
                else:
                    prompt = (
                        f"Write a news headline in the style of {author_id}:\n\n"
                        f"Article: {article[:2000]}\n\nHeadline:"
                    )
                    headline = self._generate(self.lora_model, prompt)
            else:
                headline = f"[Unknown method: {method}]"

        except Exception as e:
            headline = f"[Error: {str(e)[:100]}]"

        latency_ms = int((time.time() - t0) * 1000)

        return {
            "headline": headline,
            "method": method,
            "author_id": author_id,
            "latency_ms": latency_ms,
        }


# ── FastAPI Web Endpoint ──────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel

web_app = FastAPI(title="StyleVector LLM", version="1.0")


class GenerateRequest(BaseModel):
    prompt: str                         # article text
    method: str = "no_personalization"
    author_id: str = ""
    publication: str = ""
    max_tokens: int = 60


class HealthResponse(BaseModel):
    status: str
    model: str
    methods: list[str]


@app.function(image=llm_image)
@modal.asgi_app()
def api():
    return web_app


@web_app.get("/health")
async def health():
    return HealthResponse(
        status="ok",
        model=BASE_MODEL,
        methods=["no_personalization", "rag_bm25", "stylevector", "cold_start_sv", "lora_finetuned"],
    )


@web_app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    result = await StyleVectorLLM().generate.remote.aio(
        article=req.prompt,
        method=req.method,
        author_id=req.author_id,
        publication=req.publication,
    )
    return result


# ── Local test entrypoint ─────────────────────────────────────────────────────
@app.local_entrypoint()
def test():
    print("Testing StyleVector LLM endpoint...")

    sample_article = (
        "Scientists at MIT have developed a new battery technology that can "
        "charge electric vehicles in under five minutes, potentially "
        "revolutionizing the EV industry and reducing range anxiety among consumers."
    )

    llm = StyleVectorLLM()

    for method in ["no_personalization", "rag_bm25", "stylevector", "cold_start_sv", "lora_finetuned"]:
        print(f"\n--- {method} ---")
        result = llm.generate.remote(
            article=sample_article,
            method=method,
            author_id="aishwarya_faraswal",
            publication="TOI",
        )
        print(f"  Headline: {result['headline']}")
        print(f"  Latency:  {result['latency_ms']}ms")

    print("\n✓ All methods tested!")
