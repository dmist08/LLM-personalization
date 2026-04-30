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

import modal
import json
from pathlib import Path

# ── App + Image ───────────────────────────────────────────────────────────────
app = modal.App("stylevector-llm")

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

        # Commit volume cache after HF downloads
        hf_cache_vol.commit()
        print("[STARTUP] Ready to serve requests!")

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

        # Clean trailing garbage (steering can cause JSON metadata bleed)
        for stop in ["\n", '", "', '", "date"', '", "slug"', '", "url"', " Category:", " Source", " #", "  "]:
            idx = text.find(stop)
            if idx > 5:
                text = text[:idx]
        return text.strip().strip('"\'')

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
                return self._generate(self.base_model, prompt)
        else:
            if author_id not in self.style_vectors:
                prompt = AGNOSTIC_PROMPT.format(article=article[:2000])
                return self._generate(self.base_model, prompt)
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
        layer = self.base_model.model.layers[BEST_LAYER]
        hook_handle = layer.register_forward_hook(steering_hook)

        try:
            prompt = AGNOSTIC_PROMPT.format(article=article[:2000])
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
                headline = self._generate(self.base_model, prompt)

            elif method == "rag_bm25":
                # RAG-style: add author context to prompt
                prompt = (
                    f"Write a news headline in the style of {author_id} "
                    f"from {publication}.\n\n"
                    f"Article: {article[:2000]}\n\nHeadline:"
                )
                headline = self._generate(self.base_model, prompt)

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