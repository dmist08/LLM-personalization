# Deployment & Testing Plan — Cold-Start StyleVector
## Final Merged Plan (v3)

**Decision basis:** Analysis of both DEPLOYMENT_PLAN.md (GPU-first) and implementation_plan.md (cache-first)  
**Last updated:** 2026-04-01

---

## 1. Decision Log — What Each Plan Got Right and Wrong

| Claim | Source | Verdict | Reason |
|---|---|---|---|
| HF Spaces can run gunicorn | Both plans | ❌ Wrong | HF Spaces forces port 7860, single process, sandboxed — can't prove multi-user concurrency |
| Modal Labs for live GPU inference | GPU-first plan | ✅ Architecturally correct | But overkill for a demo that already has pre-computed results |
| Render + gunicorn for backend | Cache-first plan | ✅ Correct | 750 free hours, real Docker, gunicorn works, UptimeRobot keeps alive |
| TF-IDF fallback for new articles | Cache-first plan | ❌ Wrong | Returns cached result dressed as generation. Dishonest if professor asks |
| Groq API as live fallback | Neither plan | ✅ New, correct | Free, no credit card, LLaMA-3.1-8B, 14,400 req/day, <1s response |
| Colab + ngrok for live model | GPU-first plan | ❌ Wrong for multi-user | Crashes under 3+ concurrent users, disconnects after 30min |
| Locust + pytest testing | Cache-first plan | ✅ Correct | Right tools, right levels |
| Response correctness in concurrency test | Cache-first plan | ❌ Incomplete | Tests 200 status but not that different users get different results |
| Activation hook isolation | GPU-first plan | ✅ Real concern | But irrelevant here — no live model at serve time |
| pre-computed cache for StyleVector/Cold-Start | Both plans | ✅ Correct | Can't do activation steering live on free CPU |

---

## 2. The Problem Statement (Precise)

The professor said: **"You should be able to manage multiple requests on the server, several users might run at once."**

This means:
1. Your server must accept multiple simultaneous HTTP requests without crashing or serializing incorrectly
2. Each user must get the correct response (not another user's result)
3. You must prove this with evidence (not just claim it)

What it does **NOT** mean:
- Run 5 LLaMA instances in parallel (unrealistic, not required)
- Zero latency under load (unrealistic for a student project)
- Infinite scalability

**The correct architectural answer:** Gunicorn + async FastAPI workers, pre-computed cache as the data layer (read-only, no locks), Groq API for live fallback on new inputs. Prove correctness with pytest. Prove load with Locust.

---

## 3. Why Each Option Was Rejected

### ❌ HuggingFace Spaces (Docker)
Cannot run gunicorn. HF Spaces is sandboxed — it forces port 7860 and single process. You cannot demonstrate proper multi-worker concurrency architecture. Cold starts of 30-60s destroy a live demo.

### ❌ Modal Labs (Primary Backend)
Architecturally correct for live GPU inference, but wrong for this scenario. Your results are pre-computed. Adding Modal to serve a JSON cache is engineering theater. Use it only if you want optional live StyleVector inference in the future.

### ❌ Colab + ngrok
Single-threaded notebook. Crashes under 3+ concurrent requests. Disconnects after 30min of inactivity. Cannot be the primary demo.

### ❌ Railway
$5 trial expires after 30 days. Project needs to stay live for course evaluation window.

### ❌ Koyeb
0.1 vCPU on free tier makes it hard to demonstrate meaningful concurrency. Render is strictly better.

### ❌ TF-IDF Fallback for Unknown Articles
This is the critical issue in the original plan. TF-IDF "fallback" doesn't generate anything — it finds the most similar cached article and returns its pre-computed headline. If the professor types a new article they wrote themselves, the system silently returns a cached result from a different article. If anyone asks "did you actually generate this?", the answer is no. This is dishonest and unnecessary given Groq exists.

---

## 4. Final Architecture — The Correct Solution

```
┌───────────────────────────────────────────────────────────────────────┐
│                      COMPLETE SYSTEM ARCHITECTURE                      │
├───────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FRONTEND — Vercel (Free)                                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  React + Vite + CSS                                              │  │
│  │  - Author selector (from GET /authors)                          │  │
│  │  - Method selector: No Pers. | RAG | StyleVector | Cold-Start   │  │
│  │  - Article textarea (paste any article)                         │  │
│  │  - Generate button → POST /predict                              │  │
│  │  - Side-by-side: all 4 methods shown together                   │  │
│  │  - Shows: headline + ROUGE-L score + latency                    │  │
│  │  - Badge: "🔴 CACHED" or "🟢 LIVE (Groq)" per result           │  │
│  └──────────────────────────┬───────────────────────────────────────┘ │
│                              │ HTTPS POST (CORS enabled)               │
│                              ▼                                          │
│  BACKEND — Render (Free, 750h/month)                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Docker container                                                │  │
│  │  Gunicorn: 2 × UvicornWorker (2 independent processes)          │  │
│  │                                                                  │  │
│  │  Startup (lifespan):                                             │  │
│  │    - Load cached_predictions.json into RAM (~5-20MB)            │  │
│  │    - Load author_metadata.json into RAM (~50KB)                  │  │
│  │    - Initialize Groq client (env var: GROQ_API_KEY)             │  │
│  │                                                                  │  │
│  │  Routes:                                                         │  │
│  │    GET  /health           → status, uptime, n_authors            │  │
│  │    GET  /authors          → list of journalists                  │  │
│  │    GET  /methods          → list of methods + descriptions       │  │
│  │    POST /predict          → see routing logic below              │  │
│  │    POST /predict_batch    → max 10 per batch                     │  │
│  │    GET  /stats            → aggregate metrics per method         │  │
│  └──────────────────────────┬───────────────────────────────────────┘ │
│                              │                                          │
│           ┌──────────────────┴─────────────────┐                       │
│           ▼                                     ▼                       │
│  CACHE (pre-computed)                  GROQ API (live fallback)         │
│  ┌────────────────────────┐            ┌───────────────────────────┐   │
│  │ cached_predictions.json │           │ LLaMA-3.1-8B-Instant      │  │
│  │ {author_id: {          │           │ Free tier: 14,400 req/day  │  │
│  │   method: {            │           │ 30,000 tokens/minute       │  │
│  │     article_id: {      │           │ ~0.5 second response       │  │
│  │       predicted: "...", │          │ No credit card required    │  │
│  │       rouge_l: 0.041   │           │                            │  │
│  │     }                  │           │ Used ONLY for:             │  │
│  │   }                    │           │ - Unknown articles         │  │
│  │ }}                     │           │ - no_personalization only  │  │
│  └────────────────────────┘           └───────────────────────────┘   │
│                                                                         │
│  KEEP-ALIVE — UptimeRobot (Free)                                        │
│  Pings GET /health every 10 minutes → Render never sleeps               │
│                                                                         │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 5. The Prediction Routing Logic (Core Design Decision)

This is the logic inside `POST /predict`. It determines what each user sees and needs careful design.

```
POST /predict
  {author_id, article_text, method}
  
  Step 1: Is this a known article from a known author?
    → Hash the article_text (first 200 chars) and look up in cache
    
  Step 2a: CACHE HIT (known article, known author)
    → Return cached result for all 4 methods
    → Response badge: "CACHED (pre-computed research result)"
    → This is the main path for all evaluation
    
  Step 2b: CACHE MISS (new/unknown article)
    → Method determines behavior:
    
    "no_personalization":
      → Call Groq API with generic prompt
      → Real LLM inference, <1s
      → Badge: "LIVE (Groq LLaMA-3.1-8B)"
      → DO NOT cache the result (avoid polluting research cache)
      
    "rag_bm25":
      → If author is known: retrieve from author's cached articles via BM25
        and call Groq API with few-shot prompt
      → If author is unknown: fall back to no_personalization via Groq
      → Badge: "LIVE (Groq + BM25 retrieval)"
      
    "stylevector" or "cold_start_sv":
      → These CANNOT be computed live (require your fine-tuned model + activation steering)
      → Return clear message: "StyleVector requires pre-computed activation vectors.
         This article was not in our evaluation set. Showing closest-match result."
      → Return the most thematically similar cached article's headline
        AND clearly label it as such
      → DO NOT pretend this is a generated result
      → This is honest — the professor will appreciate the clarity
```

**Why this is correct:** You're transparent about what's pre-computed vs. live. The professor sees that your research methods (StyleVector, Cold-Start) are evaluated on a specific test set, which is standard ML practice. The live fallback (Groq) demonstrates the API works with new inputs. This is academically honest.

---

## 6. Concurrency Architecture Details

### Why Gunicorn + 2 UvicornWorkers is the Right Choice

```
Gunicorn (process manager)
├── Worker 1 (Python process, ~200MB RAM)
│   └── Uvicorn ASGI event loop
│       ├── Request A from User 1 → async handler → cache lookup → return
│       ├── Request B from User 2 → async handler → Groq API call → await → return
│       └── Request C from User 3 → async handler → cache lookup → return
│
└── Worker 2 (Python process, ~200MB RAM)
    └── Uvicorn ASGI event loop
        ├── Request D from User 4 → async handler → cache lookup → return
        └── Request E from User 5 → async handler → cache lookup → return
```

**Why 2 workers, not more:**
Render free tier = 0.1 vCPU, 512MB RAM. The cached JSON is ~10-20MB. With `--preload`, it's loaded once and shared copy-on-write. Two workers × ~200MB each = 400MB, just under the 512MB limit. 3 workers would OOM.

**Why async routes, not sync:**
The Groq API call is I/O-bound (network call). With `async def` + `await`, while Worker 1 waits for Groq to respond, its event loop can serve other cached requests. This is why you get hundreds of concurrent connections from 2 processes — most requests (cache hits) complete in <5ms.

**Dockerfile CMD:**
```dockerfile
CMD ["gunicorn", "backend.app:app",
     "--workers", "2",
     "--worker-class", "uvicorn.workers.UvicornWorker",
     "--bind", "0.0.0.0:10000",
     "--preload",
     "--timeout", "120",
     "--keepalive", "5",
     "--access-logfile", "-"]
```

**Why `--preload`:** Loads the JSON cache once in the master process before forking workers. Workers inherit it via copy-on-write. Without `--preload`, each worker loads its own copy — wastes 2× RAM and doubles startup time.

**Why `--timeout 120`:** Groq API calls are usually <1s but can take up to 5s under load. The 120s timeout gives plenty of headroom before killing stuck requests.

---

## 7. File Structure

```
DL/
├── backend/
│   ├── app.py                 ← FastAPI + lifespan + all routes
│   ├── schemas.py             ← Pydantic v2 request/response models
│   ├── cache.py               ← Cache loading + article matching logic
│   ├── groq_client.py         ← Groq API wrapper + retry logic
│   ├── requirements.txt       ← fastapi, uvicorn, gunicorn, pydantic, groq, scikit-learn
│   └── Dockerfile             ← python:3.10-slim, non-root, health check
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx            ← Main component
│   │   ├── components/
│   │   │   ├── AuthorSelect.jsx
│   │   │   ├── MethodSelect.jsx
│   │   │   ├── ArticleInput.jsx
│   │   │   └── ResultCard.jsx   ← Shows headline + badge (CACHED/LIVE) + score
│   │   └── api.js             ← fetch wrapper pointing to Render URL
│   ├── .env.local             ← VITE_API_URL=https://your-app.onrender.com
│   ├── package.json
│   └── vite.config.js
│
├── scripts/
│   └── prepare_deployment_cache.py   ← Builds cached_predictions.json from outputs/
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py            ← fixtures: test client, mock cache, sample data
│   ├── test_api_health.py     ← health, authors, methods endpoints
│   ├── test_api_predict.py    ← predict: known author, unknown author, invalid input
│   ├── test_api_batch.py      ← batch predict, size limits
│   ├── test_api_validation.py ← Pydantic: too-short article, bad method, bad author
│   ├── test_concurrency.py    ← asyncio.gather correctness + uniqueness test
│   └── locustfile.py          ← load test: 50 users, mixed traffic
│
├── outputs/
│   └── cached_predictions.json        ← built by prepare_deployment_cache.py
│
├── render.yaml                ← Render blueprint (auto-deploy from GitHub)
└── .github/
    └── workflows/
        └── test.yml           ← pytest on push to main
```

---

## 8. Complete Testing Plan

### Level 1 — Unit Tests (pytest)

Every module tested in isolation with a mock cache (no network calls during tests).

```python
# tests/conftest.py

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

MOCK_CACHE = {
    "ananya-das": {
        "no_personalization": {
            "abc123": {
                "predicted": "India's tech sector reports record growth",
                "ground_truth": "India Tech Sector Hits New High in Q4",
                "rouge_l": 0.041,
                "article_text": "India's technology sector reported record growth..."
            }
        },
        "rag_bm25": {
            "abc123": {"predicted": "India Tech Hits Record in Fourth Quarter", "rouge_l": 0.044}
        },
        "stylevector": {
            "abc123": {"predicted": "India's Tech Sector Breaks Q4 Records: A New High", "rouge_l": 0.046}
        },
        "cold_start_sv": {
            "abc123": {"predicted": "Record Growth in India's Tech Sector: Q4 Report", "rouge_l": 0.047}
        }
    }
}

MOCK_AUTHOR_METADATA = {
    "ananya-das": {"name": "Ananya Das", "source": "HT", "total": 618, "class": "rich"},
    "utpal-parashar": {"name": "Utpal Parashar", "source": "HT", "total": 606, "class": "rich"},
}

@pytest.fixture
def test_client(monkeypatch):
    """Test client with mocked cache (no real JSON file needed)."""
    from backend.app import app
    monkeypatch.setattr("backend.cache.PREDICTIONS", MOCK_CACHE)
    monkeypatch.setattr("backend.cache.AUTHOR_METADATA", MOCK_AUTHOR_METADATA)
    return TestClient(app)

@pytest.fixture
async def async_client(monkeypatch):
    """Async test client for concurrency tests."""
    from backend.app import app
    monkeypatch.setattr("backend.cache.PREDICTIONS", MOCK_CACHE)
    monkeypatch.setattr("backend.cache.AUTHOR_METADATA", MOCK_AUTHOR_METADATA)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
```

```python
# tests/test_api_health.py

def test_health_returns_ok(test_client):
    r = test_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "n_authors" in data
    assert isinstance(data["n_authors"], int)

def test_authors_returns_list(test_client):
    r = test_client.get("/authors")
    assert r.status_code == 200
    authors = r.json()
    assert isinstance(authors, list)
    assert len(authors) > 0
    assert all("author_id" in a for a in authors)
    assert all("author_name" in a for a in authors)

def test_methods_returns_all_four(test_client):
    r = test_client.get("/methods")
    assert r.status_code == 200
    methods = r.json()
    method_ids = [m["method_id"] for m in methods]
    assert "no_personalization" in method_ids
    assert "rag_bm25" in method_ids
    assert "stylevector" in method_ids
    assert "cold_start_sv" in method_ids
```

```python
# tests/test_api_predict.py

KNOWN_ARTICLE = "India's technology sector reported record growth in the fourth quarter..."

def test_predict_known_author_known_article(test_client):
    """Cache hit: should return pre-computed headline."""
    r = test_client.post("/predict", json={
        "author_id": "ananya-das",
        "article_text": KNOWN_ARTICLE,
        "method": "stylevector"
    })
    assert r.status_code == 200
    data = r.json()
    assert data["headline"]
    assert len(data["headline"].split()) >= 3
    assert data["source"] == "cached"
    assert data["rouge_l"] > 0

def test_predict_unknown_article_no_personalization(test_client, monkeypatch):
    """Cache miss on no_personalization: should call Groq and return live result."""
    FAKE_GROQ_HEADLINE = "New Vaccine Shows 90% Efficacy in Phase 3 Trial"
    
    async def mock_groq_call(article_text, prompt):
        return FAKE_GROQ_HEADLINE
    
    monkeypatch.setattr("backend.groq_client.generate_headline", mock_groq_call)
    
    r = test_client.post("/predict", json={
        "author_id": "ananya-das",
        "article_text": "A completely new article about vaccine trials never seen before...",
        "method": "no_personalization"
    })
    assert r.status_code == 200
    data = r.json()
    assert data["headline"] == FAKE_GROQ_HEADLINE
    assert data["source"] == "live_groq"

def test_predict_unknown_article_stylevector_is_honest(test_client):
    """StyleVector on unknown article: must NOT pretend to generate."""
    r = test_client.post("/predict", json={
        "author_id": "ananya-das",
        "article_text": "A completely new unseen article text goes here for testing purposes...",
        "method": "stylevector"
    })
    assert r.status_code == 200
    data = r.json()
    # Must NOT return a fake generated result
    assert data["source"] in ("cached_nearest", "unavailable")
    # Must include an explanation
    assert "note" in data or "message" in data
```

```python
# tests/test_api_validation.py

def test_article_too_short_rejected(test_client):
    r = test_client.post("/predict", json={
        "author_id": "ananya-das",
        "article_text": "Short",   # Way under 50 char minimum
        "method": "no_personalization"
    })
    assert r.status_code == 422   # Pydantic validation error

def test_invalid_method_rejected(test_client):
    r = test_client.post("/predict", json={
        "author_id": "ananya-das",
        "article_text": "A" * 100,
        "method": "nonexistent_method"
    })
    assert r.status_code == 422

def test_batch_over_limit_rejected(test_client):
    """Batch requests above 10 should be rejected."""
    requests = [
        {"author_id": "ananya-das", "article_text": "A" * 100, "method": "no_personalization"}
    ] * 11   # 11 items — over the limit
    r = test_client.post("/predict_batch", json={"requests": requests})
    assert r.status_code == 422

def test_empty_author_rejected(test_client):
    r = test_client.post("/predict", json={
        "author_id": "",
        "article_text": "A" * 100,
        "method": "no_personalization"
    })
    assert r.status_code == 422
```

---

### Level 2 — Concurrency Tests (pytest + asyncio)

This is the critical test for the professor's requirement. It proves TWO things:
1. All requests complete successfully under concurrent load
2. Each user's request returns the correct, unique result (not swapped)

```python
# tests/test_concurrency.py

import pytest
import asyncio
import pytest_asyncio

# Required: pip install pytest-asyncio httpx
# In pytest.ini: asyncio_mode = auto

@pytest.mark.asyncio
async def test_20_concurrent_requests_all_succeed(async_client):
    """
    Fire 20 simultaneous POST /predict requests.
    ALL must succeed (200 status).
    This proves the server doesn't crash under concurrent load.
    """
    payload = {
        "author_id": "ananya-das",
        "article_text": "India's technology sector reported record growth...",
        "method": "no_personalization"
    }
    
    tasks = [async_client.post("/predict", json=payload) for _ in range(20)]
    responses = await asyncio.gather(*tasks, return_exceptions=False)
    
    status_codes = [r.status_code for r in responses]
    assert all(code == 200 for code in status_codes), \
        f"Expected all 200, got: {status_codes}"

@pytest.mark.asyncio
async def test_different_authors_get_different_headlines(async_client):
    """
    THE CRITICAL TEST.
    Two users request headlines for different authors simultaneously.
    They must NOT receive each other's results.
    
    This proves:
    1. No shared mutable state between requests
    2. Author routing works correctly under concurrent access
    """
    payload_ananya = {
        "author_id": "ananya-das",
        "article_text": "India's technology sector reported record growth...",
        "method": "stylevector"
    }
    payload_utpal = {
        "author_id": "utpal-parashar",
        "article_text": "India's technology sector reported record growth...",
        "method": "stylevector"  # Same article, different authors → must get different headlines
    }
    
    task_ananya = async_client.post("/predict", json=payload_ananya)
    task_utpal = async_client.post("/predict", json=payload_utpal)
    
    # Fire simultaneously
    r_ananya, r_utpal = await asyncio.gather(task_ananya, task_utpal)
    
    assert r_ananya.status_code == 200
    assert r_utpal.status_code == 200
    
    headline_ananya = r_ananya.json()["headline"]
    headline_utpal = r_utpal.json()["headline"]
    
    # Different authors → different style vectors → different headlines
    assert headline_ananya != headline_utpal, \
        "CRITICAL FAILURE: Two different authors returned the same headline. " \
        "This means either the cache routing is broken or responses were swapped."
    
    # Both must be non-empty and reasonable length
    assert len(headline_ananya.split()) >= 3, "Ananya's headline is too short"
    assert len(headline_utpal.split()) >= 3, "Utpal's headline is too short"

@pytest.mark.asyncio
async def test_batch_concurrent_with_mixed_authors(async_client):
    """
    10 concurrent requests, each for a different author.
    All must succeed. Response author_ids must match request author_ids.
    """
    authors = ["ananya-das", "utpal-parashar"] * 5  # 10 requests
    
    async def make_request(author_id: str):
        r = await async_client.post("/predict", json={
            "author_id": author_id,
            "article_text": "India's technology sector reported record growth...",
            "method": "no_personalization"
        })
        return {"status": r.status_code, "author_id": r.json().get("author_id"), 
                "requested": author_id}
    
    results = await asyncio.gather(*[make_request(a) for a in authors])
    
    for result in results:
        assert result["status"] == 200
        # Response must confirm which author it was for
        assert result["author_id"] == result["requested"], \
            f"Author mismatch: requested {result['requested']}, got {result['author_id']}"

@pytest.mark.asyncio
async def test_health_survives_concurrent_predict_load(async_client):
    """
    While 10 predict requests are in flight, health endpoint must still respond.
    This proves the server isn't completely locked under load.
    """
    predict_tasks = [
        async_client.post("/predict", json={
            "author_id": "ananya-das",
            "article_text": "India's technology sector reported record growth...",
            "method": "no_personalization"
        }) for _ in range(10)
    ]
    health_task = async_client.get("/health")
    
    all_tasks = predict_tasks + [health_task]
    responses = await asyncio.gather(*all_tasks)
    
    # Health response is the last one
    health_response = responses[-1]
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"
```

---

### Level 3 — Load Tests (Locust)

Run this against the live Render deployment. Screenshot the dashboard for the professor.

```python
# tests/locustfile.py

import random
from locust import HttpUser, task, between

# Realistic test data — use actual known authors from your dataset
KNOWN_AUTHORS = [
    "ananya-das", "utpal-parashar", "rasesh-mandani",
    "yash-nitish-bajaj", "neeshita-nyayapati", "vishal-mathur",
    "sehjal-gupta", "bharti-jain", "vishwa-mohan", "amit-kumar"
]

SAMPLE_ARTICLES = [
    # Use the first 200 chars of real articles from your test set
    # These are guaranteed cache hits → fast response
    "India's technology sector reported record growth in the fourth quarter of the fiscal year...",
    "The Supreme Court today delivered a landmark judgment on data privacy rights of citizens...",
    "The government announced new agricultural reforms targeting rural farmers in northern states...",
    "Scientists at IIT Bombay have developed a new approach to solar energy storage systems...",
    "India's cricket team announced a new selection policy ahead of the upcoming World Cup series...",
    # Add 1-2 genuinely new articles to test Groq live path
    "Breaking news article not in the training set about a completely novel topic today...",
]

METHODS = ["no_personalization", "rag_bm25", "stylevector", "cold_start_sv"]

class HeadlineAPIUser(HttpUser):
    """
    Simulates a real user of the demo application.
    Users spend 1-3 seconds between requests (realistic think time).
    """
    wait_time = between(1, 3)
    
    @task(1)
    def check_health(self):
        """Simulates monitoring / keep-alive checks."""
        with self.client.get("/health", catch_response=True) as r:
            if r.status_code == 200:
                r.success()
            else:
                r.failure(f"Health check failed: {r.status_code}")
    
    @task(2)
    def browse_authors(self):
        """Simulates user exploring available journalists."""
        with self.client.get("/authors", catch_response=True) as r:
            if r.status_code == 200:
                data = r.json()
                if not isinstance(data, list) or len(data) == 0:
                    r.failure("Authors list is empty or malformed")
                else:
                    r.success()
            else:
                r.failure(f"Authors endpoint failed: {r.status_code}")
    
    @task(5)  # Majority of traffic is headline generation
    def generate_headline(self):
        """Simulates user generating a headline — main workflow."""
        payload = {
            "author_id": random.choice(KNOWN_AUTHORS),
            "article_text": random.choice(SAMPLE_ARTICLES),
            "method": random.choice(METHODS)
        }
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            timeout=30     # Cache hits <100ms, Groq calls <5s — 30s is very generous
        ) as r:
            if r.status_code == 200:
                data = r.json()
                if not data.get("headline") or len(data["headline"]) < 3:
                    r.failure(f"Empty or too-short headline: '{data.get('headline', '')}'")
                elif data.get("author_id") != payload["author_id"]:
                    r.failure(f"Author mismatch: requested {payload['author_id']}, got {data.get('author_id')}")
                else:
                    r.success()
            elif r.status_code == 422:
                r.failure("Validation error — test data may be malformed")
            elif r.status_code == 429:
                r.failure("Rate limit hit (Groq API)")
            else:
                r.failure(f"Unexpected status: {r.status_code} — {r.text[:200]}")
    
    @task(1)
    def batch_predict(self):
        """Simulates batch comparison workflow."""
        author_id = random.choice(KNOWN_AUTHORS)
        article = random.choice(SAMPLE_ARTICLES)
        payload = {
            "requests": [
                {"author_id": author_id, "article_text": article, "method": m}
                for m in METHODS
            ]
        }
        with self.client.post("/predict_batch", json=payload, catch_response=True) as r:
            if r.status_code == 200:
                results = r.json()
                if len(results) != 4:
                    r.failure(f"Expected 4 results, got {len(results)}")
                else:
                    r.success()
            else:
                r.failure(f"Batch predict failed: {r.status_code}")
```

**Run commands:**
```bash
# Visual dashboard (for professor demo)
locust -f tests/locustfile.py --host https://your-app.onrender.com
# Open http://localhost:8089 → set 50 users, spawn rate 5/s → Start

# Headless run (for CI / automated evidence)
locust -f tests/locustfile.py \
    --host https://your-app.onrender.com \
    --headless \
    --users 50 \
    --spawn-rate 5 \
    --run-time 60s \
    --html outputs/testing/load_test_50users.html \
    --csv outputs/testing/load_test_50users

# Against local server (use this during development)
uvicorn backend.app:app --workers 1 --port 10000  # dev server
locust -f tests/locustfile.py --host http://localhost:10000 --headless -u 20 -r 2 --run-time 30s
```

**Target metrics:**
| Metric | Minimum Pass | Strong Pass |
|---|---|---|
| Users | 20 | 50 |
| Failure rate | < 5% | 0% |
| P50 response time | < 200ms | < 100ms |
| P95 response time | < 2000ms | < 500ms |
| Requests/sec | > 5 | > 20 |

Cache hits (95% of traffic) will be <50ms. The Groq live path adds ~500-1000ms for the 5% of unknown articles. This gives strong metrics.

---

## 9. Backend Implementation Spec

### `backend/app.py` — Complete API

```python
# backend/app.py — Key structure (full implementation in Prompt 12)

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import time

from backend.schemas import PredictRequest, PredictResponse, BatchRequest
from backend.cache import load_cache, lookup_prediction
from backend.groq_client import generate_headline_live

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all data at startup. Fail fast if cache missing."""
    logger.info("Loading pre-computed predictions cache...")
    start = time.time()
    load_cache()   # Loads into module-level globals in cache.py
    logger.info(f"Cache loaded in {time.time() - start:.2f}s")
    yield
    # Shutdown: nothing to clean up

app = FastAPI(
    title="Cold-Start StyleVector API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Restrict to Vercel domain in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Generate or retrieve a headline.
    Cache hit: <50ms. Groq live: ~500-1000ms.
    """
    start = time.time()
    
    result = await lookup_prediction(
        author_id=request.author_id,
        article_text=request.article_text,
        method=request.method
    )
    
    return PredictResponse(
        headline=result["headline"],
        author_id=request.author_id,
        method=request.method,
        source=result["source"],           # "cached" | "live_groq" | "cached_nearest"
        rouge_l=result.get("rouge_l"),
        note=result.get("note"),
        latency_ms=(time.time() - start) * 1000
    )
```

### `backend/cache.py` — Routing Logic

```python
# backend/cache.py — Core routing (no TF-IDF fallback — use Groq or honest fallback)

import json
import hashlib
from pathlib import Path

PREDICTIONS: dict = {}
AUTHOR_METADATA: dict = {}
ARTICLE_INDEX: dict = {}   # hash → article_id for fast lookup

def load_cache():
    """Load cached predictions at startup."""
    global PREDICTIONS, AUTHOR_METADATA, ARTICLE_INDEX
    
    cache_path = Path("outputs/cached_predictions.json")
    meta_path = Path("data/processed/indian/author_metadata.json")
    
    if not cache_path.exists():
        raise RuntimeError(f"Cache file not found: {cache_path}. Run scripts/prepare_deployment_cache.py first.")
    
    with open(cache_path) as f:
        PREDICTIONS = json.load(f)
    
    with open(meta_path) as f:
        AUTHOR_METADATA = json.load(f)
    
    # Build article hash index for O(1) lookup
    for author_id, methods in PREDICTIONS.items():
        for method, articles in methods.items():
            for article_id, data in articles.items():
                text_hash = _hash_article(data.get("article_text", ""))
                ARTICLE_INDEX[text_hash] = article_id

def _hash_article(text: str) -> str:
    """First 300 chars hash — tolerant of minor whitespace differences."""
    normalized = " ".join(text[:300].split())
    return hashlib.md5(normalized.encode()).hexdigest()

async def lookup_prediction(author_id: str, article_text: str, method: str) -> dict:
    """
    Main routing logic. Returns dict with headline + metadata.
    """
    from backend.groq_client import generate_headline_live
    
    article_hash = _hash_article(article_text)
    
    # Check: is this author known?
    if author_id not in PREDICTIONS:
        # Unknown author — can only do live Groq for no_personalization
        if method == "no_personalization":
            headline = await generate_headline_live(article_text)
            return {"headline": headline, "source": "live_groq"}
        else:
            return {
                "headline": "Unknown author. StyleVector requires pre-computed activation vectors.",
                "source": "unavailable",
                "note": f"Author '{author_id}' was not in the evaluation set."
            }
    
    # Author is known. Check: is this exact article in cache?
    article_id = ARTICLE_INDEX.get(article_hash)
    author_cache = PREDICTIONS[author_id]
    
    if article_id and method in author_cache and article_id in author_cache[method]:
        # CACHE HIT — fast path
        cached = author_cache[method][article_id]
        return {
            "headline": cached["predicted"],
            "rouge_l": cached.get("rouge_l"),
            "source": "cached",
        }
    
    # CACHE MISS — article not seen before
    if method == "no_personalization":
        headline = await generate_headline_live(article_text)
        return {"headline": headline, "source": "live_groq"}
    
    elif method == "rag_bm25":
        # Can do BM25 retrieval from author's cached articles + Groq generation
        examples = get_author_examples(author_id, n=2)
        headline = await generate_headline_live(article_text, rag_examples=examples)
        return {"headline": headline, "source": "live_groq_rag"}
    
    else:  # stylevector or cold_start_sv
        # Honest response — cannot do live activation steering
        best_match = find_closest_cached_article(author_id, article_text)
        return {
            "headline": best_match["predicted"] if best_match else "No cached prediction available",
            "source": "cached_nearest",
            "note": "StyleVector requires pre-computed activation vectors. "
                    "This article was not in the evaluation set. "
                    "Showing result from most similar cached article."
        }
```

### `backend/groq_client.py` — Live Inference

```python
# backend/groq_client.py

import os
import logging
from groq import AsyncGroq

logger = logging.getLogger(__name__)

# Initialize client once at module load (reused across requests)
_client: AsyncGroq | None = None

def get_client() -> AsyncGroq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set")
        _client = AsyncGroq(api_key=api_key)
    return _client

async def generate_headline_live(
    article_text: str,
    rag_examples: list[dict] | None = None,
    max_tokens: int = 40
) -> str:
    """
    Generate a headline using Groq's LLaMA-3.1-8B-Instant.
    Free tier: 14,400 req/day, ~30,000 tokens/minute.
    Typical latency: 300-800ms.
    """
    client = get_client()
    
    # Truncate article to save tokens (headlines don't need full article)
    truncated = " ".join(article_text.split()[:300])
    
    if rag_examples:
        # RAG: few-shot examples from author's history
        examples_text = "\n\n".join([
            f"Article: {ex['article_text'][:200]}\nHeadline: {ex['headline']}"
            for ex in rag_examples[:2]
        ])
        prompt = (
            f"Here are example headlines from this journalist:\n\n"
            f"{examples_text}\n\n"
            f"Now write a headline for:\n{truncated}\n\nHeadline:"
        )
    else:
        prompt = (
            f"Write a concise, neutral news headline for the following article. "
            f"Only output the headline, nothing else.\n\n{truncated}\n\nHeadline:"
        )
    
    try:
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Most permissive free tier limits
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3     # Low temp for consistent headline style
        )
        headline = response.choices[0].message.content.strip()
        # Clean: remove "Headline:", quotes, leading/trailing whitespace
        headline = headline.strip('"\'').removeprefix("Headline:").strip()
        return headline
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return "Headline generation temporarily unavailable"
```

---

## 10. Deployment Steps (In Order)

### Step 1 — Accounts (Do This Now, 10 Minutes)
1. Create [Render account](https://render.com) — sign up with GitHub
2. Create [Vercel account](https://vercel.com) — sign up with GitHub
3. Create [UptimeRobot account](https://uptimerobot.com) — free, no card
4. Get [Groq API key](https://console.groq.com) — free, no card. Model: `llama-3.1-8b-instant`

### Step 2 — Build the Cache (Pre-computation Complete)
```bash
# After evaluation pipeline is done (outputs/*.jsonl exist)
python scripts/prepare_deployment_cache.py

# Verify output
python -c "
import json
with open('outputs/cached_predictions.json') as f:
    cache = json.load(f)
authors = list(cache.keys())
print(f'Authors: {len(authors)}')
print(f'Sample: {authors[:3]}')
print(f'Methods for first author: {list(cache[authors[0]].keys())}')
"
```

### Step 3 — Backend Dockerfile

```dockerfile
# backend/Dockerfile

FROM python:3.10-slim

# Non-root user (security best practice)
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install dependencies first (Docker layer cache)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY outputs/cached_predictions.json ./outputs/
COPY data/processed/indian/author_metadata.json ./data/processed/indian/

# Switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10000/health')"

# Gunicorn with 2 async workers
CMD ["gunicorn", "backend.app:app",
     "--workers", "2",
     "--worker-class", "uvicorn.workers.UvicornWorker",
     "--bind", "0.0.0.0:10000",
     "--preload",
     "--timeout", "120",
     "--keepalive", "5",
     "--access-logfile", "-"]
```

**`backend/requirements.txt`:**
```
fastapi==0.115.0
uvicorn==0.30.0
gunicorn==22.0.0
pydantic==2.7.0
groq==0.9.0
scikit-learn==1.5.0   # For find_closest_cached_article (cosine similarity)
python-multipart==0.0.9
```

### Step 4 — render.yaml (Auto-Deploy)

```yaml
# render.yaml

services:
  - type: web
    name: cold-start-stylevector-api
    runtime: docker
    dockerfilePath: ./backend/Dockerfile
    envVars:
      - key: GROQ_API_KEY
        sync: false   # You'll set this manually in Render dashboard (don't commit API key)
    healthCheckPath: /health
    region: oregon
    plan: free
```

Deploy process:
1. Push to GitHub (`git push`)
2. Render auto-builds and deploys from `render.yaml`
3. Set `GROQ_API_KEY` in Render Dashboard → Environment Variables (never in code)

### Step 5 — Frontend (Vite + React)

```bash
cd frontend
npm install   # After scaffolding with: npx create-vite@latest . --template react
```

**`.env.local` (not committed to git):**
```
VITE_API_URL=https://cold-start-stylevector-api.onrender.com
```

Key UI features:
- Author dropdown (fetched from `/authors` on load)
- Method tabs: No Personalization | RAG | StyleVector | Cold-Start
- Large article textarea
- "Generate All Methods" button → 4 parallel requests → side-by-side results
- Each result card shows: headline + source badge (CACHED/LIVE) + ROUGE-L
- Optional: queue depth indicator for Groq rate limit awareness

```bash
# Deploy to Vercel
npx vercel deploy --prod
# Follow prompts → set VITE_API_URL as environment variable in Vercel dashboard
```

### Step 6 — UptimeRobot (5 Minutes)

1. Go to uptimerobot.com → Add New Monitor
2. Monitor Type: HTTP(S)
3. URL: `https://cold-start-stylevector-api.onrender.com/health`
4. Check interval: Every 10 minutes
5. Alert when down: optional email alert

This prevents Render's free tier 15-minute sleep.

### Step 7 — CI/CD (GitHub Actions)

```yaml
# .github/workflows/test.yml

name: Test Suite

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      
      - name: Install dependencies
        run: |
          pip install fastapi uvicorn pydantic groq scikit-learn \
                      pytest pytest-asyncio httpx
      
      - name: Create mock cache for testing
        run: |
          mkdir -p outputs data/processed/indian
          python tests/create_mock_cache.py   # Script that writes minimal mock JSON
      
      - name: Run unit and concurrency tests
        run: |
          pytest tests/ -v --tb=short \
            --ignore=tests/locustfile.py \
            -k "not slow"
        
      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results
          path: test-results/
```

---

## 11. What to Show the Professor

Prepare these 5 things before the demo:

**1. Live demo URL (Vercel frontend)**
Open the frontend. Select a journalist. Paste an article. Click generate. Show all 4 methods side-by-side. Point out: "CACHED" results are from our evaluation run, "LIVE" results are real-time Groq inference.

**2. Locust dashboard screenshot**
```bash
locust -f tests/locustfile.py --host https://your-app.onrender.com
# Set 50 users, 5/s spawn rate, let run 60s
# Screenshot: shows concurrent users, RPS, 0% failure rate, <500ms P95
```

**3. pytest output (all green)**
```bash
pytest tests/ -v --tb=short --ignore=tests/locustfile.py
# Screenshot: all green, including test_concurrency.py
# Key test: test_different_authors_get_different_headlines
```

**4. Architecture diagram (from this document)**
Print or present the ASCII architecture diagram in Section 4.

**5. UptimeRobot dashboard**
Shows 30-day uptime history. Proves the API has been live and available, not just working during the demo.

---

## 12. Cost Summary

| Component | Platform | Cost | Notes |
|---|---|---|---|
| Backend API | Render | $0 | 750 free hours/month. UptimeRobot prevents sleep |
| Frontend | Vercel | $0 | 100GB bandwidth, unlimited deploys |
| Live inference (fallback) | Groq | $0 | 14,400 req/day, 500K tokens/day free |
| Keep-alive pings | UptimeRobot | $0 | 50 free monitors, 10min intervals |
| Load testing | Locust (local) | $0 | Runs on your laptop |
| Total | — | **$0** | |

---

## 13. Failure Mode and Backup Plans

| Failure | Cause | Fix |
|---|---|---|
| Render sleeps mid-demo | UptimeRobot misconfigured | Ensure monitor is active; hit API manually to wake |
| Groq API 429 rate limit | >30 requests/minute | Cache unknown articles after first Groq call (simple dict in memory) |
| Large cache JSON crashes 512MB RAM | Cache > 400MB | Lazy-load per-author instead of all at once (or split into author-sharded files) |
| Vercel CORS error | Wrong domain in allow_origins | Add Vercel URL to CORS origins list in app.py |
| CI test fails (no cache file) | GitHub Actions | Add `create_mock_cache.py` script that generates minimal test fixture |

### If Render Completely Fails
Deploy the same Docker image to HuggingFace Spaces (no gunicorn, single process — acceptable as backup, cannot run full load test but demo still works).
```bash
# HF Spaces deployment — backup only
# Change port to 7860 in CMD, remove --workers flag, push to HF Hub
```
