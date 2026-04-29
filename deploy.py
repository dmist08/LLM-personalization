import modal
from fastapi import FastAPI
from pydantic import BaseModel

# ── App & image ────────────────────────────────────────────────
app = modal.App("my-custom-llm-api")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "huggingface_hub", "fastapi", "uvicorn")
)

# ── Persistent cache ───────────────────────────────────────────
volume = modal.Volume.from_name("llm-weights", create_if_missing=True)
MODEL_CACHE = "/model-cache"
HF_MODEL_ID = "Khushali1305/my-custom-llm"


# ── Step A: Download weights once ─────────────────────────────
@app.function(
    image=vllm_image,
    volumes={MODEL_CACHE: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def download_model():
    import os
    from huggingface_hub import snapshot_download
    print(f"Downloading {HF_MODEL_ID}...")
    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=f"{MODEL_CACHE}/{HF_MODEL_ID}",
        token=os.environ["HF_TOKEN"],
    )
    volume.commit()
    print("Model cached in Modal Volume!")


# ── Step B: Inference class ────────────────────────────────────
@app.cls(
    image=vllm_image,
    gpu="T4",
    volumes={MODEL_CACHE: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    timeout=600,
)
class LLM:
    @modal.enter()
    def load_model(self):
        from vllm import LLM as vLLM          # ✅ Fix 2: aliased to avoid name clash
        print("Loading model into GPU...")
        self.model = vLLM(model=f"{MODEL_CACHE}/{HF_MODEL_ID}")
        print("Model ready!")

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 80, temperature: float = 0.7) -> str:
        from vllm import SamplingParams        # ✅ Fix 3: per-request SamplingParams
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["\n", "Headline:", "Article:"],
        )
        outputs = self.model.generate([prompt], sampling_params)
        if not outputs or not outputs[0].outputs:
            return ""
        text = outputs[0].outputs[0].text.strip()
        for stop in ["\n", "Headline:", "Article:"]:
            text = text.split(stop)[0].strip()
        return text


# ── Prompt builder ─────────────────────────────────────────────
def build_prompt(source_text: str, method: str, author_id: str, publication: str) -> str:
    base = f"Article: {source_text.strip()}\nHeadline:"

    if method == "no_personalization":
        return base

    pub_line    = f"Publication: {publication}" if publication else ""
    author_line = f"Author style: {author_id}"  if author_id  else ""

    if method == "rag_bm25":
        return f"{pub_line}\n{base}"

    if method in ("stylevector", "cold_start_sv"):
        return (
            f"Write a headline in the style of {author_id} for {publication}.\n"
            f"{pub_line}\n{author_line}\n{base}"
        )

    return base


# ── Step C: REST API endpoint ──────────────────────────────────
web_app = FastAPI()


class PromptRequest(BaseModel):
    prompt: str
    method: str        = "no_personalization"
    author_id: str     = ""
    publication: str   = ""
    max_tokens: int    = 80
    temperature: float = 0.7


@app.function(image=vllm_image)
@modal.asgi_app()
def api():
    return web_app


@web_app.post("/generate")
async def generate_endpoint(req: PromptRequest):
    instruction_prompt = build_prompt(
        source_text=req.prompt,
        method=req.method,
        author_id=req.author_id,
        publication=req.publication,
    )

    # ✅ Fix 1 & 4: async call, no manual instantiation
    result = await LLM().generate.remote.aio(
        instruction_prompt,
        req.max_tokens,
        req.temperature,
    )

    if not result:
        return {
            "headline": "",
            "model":    HF_MODEL_ID,
            "method":   req.method,
            "error":    "Model returned empty output — check: modal app logs my-custom-llm-api",
        }

    return {
        "headline":    result,
        "model":       HF_MODEL_ID,
        "method":      req.method,
        "author_id":   req.author_id,
        "publication": req.publication,
    }


@web_app.get("/health")
async def health():
    return {"status": "ok", "model": HF_MODEL_ID}