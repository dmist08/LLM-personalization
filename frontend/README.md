# StyleVector — Cold-Start Newsroom

> Personalized headline generation for Indian journalists.
> DA-IICT · Deep Learning IT549 · 2026

---

## Stack

| Layer      | Tech                                   |
|------------|----------------------------------------|
| Frontend   | React 18 + Vite + TailwindCSS          |
| Backend    | Flask 3 + Flask-CORS                   |
| Database   | MongoDB Atlas                          |
| AI Model   | Your LLM deployed on Modal             |
| Deploy FE  | Vercel                                 |
| Deploy BE  | Modal (serverless Python)              |

---

## Project Structure

```
stylevector/
├── frontend/
│   ├── src/
│   │   ├── pages/          LoginPage, HomePage, ChatPage
│   │   ├── components/     Sidebar, InputBar, HeadlineCard
│   │   ├── services/       api.js  ← all HTTP calls
│   │   └── context/        ThemeContext, AuthContext
│   ├── vercel.json
│   └── .env.example
│
└── backend/
    ├── app.py              Flask factory
    ├── modal_app.py        Modal deploy entry
    ├── seed_db.py          Seed 43 journalists into MongoDB
    ├── routes/
    │   ├── generate.py     POST /api/generate
    │   ├── authors.py      GET  /api/authors
    │   └── history.py      GET/DELETE /api/history
    ├── db/
    │   └── mongo.py        MongoDB singleton
    └── .env.example
```

---

## Local Development

### 1. Prerequisites

- Node.js 18+
- Python 3.11+
- MongoDB Atlas free cluster → get your URI
- Modal account → `pip install modal && modal setup`

---

### 2. Clone & install

```bash
# Frontend
cd stylevector/frontend
cp .env.example .env          # edit if needed
npm install
npm run dev                   # → http://localhost:5173

# Backend (new terminal)
cd stylevector/backend
cp .env.example .env          # fill MONGODB_URI + MODAL_LLM_URL
pip install -r requirements.txt
python app.py                 # → http://localhost:5000
```

---

### 3. Seed the database

```bash
cd backend
python seed_db.py
# ✅ 43 inserted, indexes created
```

---

## Deployment

### Step 1 — Deploy your LLM to Modal

```bash
# Inside your LLM project folder
modal deploy your_llm_app.py
# → Copy the printed URL, e.g.:
#   https://your-org--your-llm-generate.modal.run
```

Update `backend/.env`:
```
MODAL_LLM_URL=https://your-org--your-llm-generate.modal.run
```

---

### Step 2 — Set Modal secrets for the Flask backend

```bash
# Run once — stores your secrets in Modal's vault
modal secret create stylevector-secrets \
  MONGODB_URI="mongodb+srv://..." \
  MONGODB_DB="stylevector" \
  MODAL_LLM_URL="https://your-org--your-llm-generate.modal.run" \
  ALLOWED_ORIGINS="https://your-app.vercel.app"
```

---

### Step 3 — Deploy Flask backend to Modal

```bash
cd backend
modal deploy modal_app.py
# → Printed URL will look like:
#   https://your-org--stylevector-backend-flask-app.modal.run
```

Test it:
```bash
curl https://your-org--stylevector-backend-flask-app.modal.run/api/health
# {"status":"ok","service":"stylevector-backend"}
```

---

### Step 4 — Deploy frontend to Vercel

```bash
cd frontend
# Push to GitHub first, then:
vercel --prod
# OR connect repo at https://vercel.com/new
```

Set these environment variables in Vercel dashboard:

| Key             | Value                                                          |
|-----------------|----------------------------------------------------------------|
| `VITE_API_URL`  | `https://your-org--stylevector-backend-flask-app.modal.run/api` |

---

## API Reference

### `GET /api/health`
Returns `{ status: "ok" }`.

### `GET /api/authors?publication=TOI`
Returns journalist list. `publication` is optional (returns all 43 if omitted).

### `POST /api/generate`
```json
{
  "source_text":  "Full article body...",
  "author_id":    "toi_ps",
  "publication":  "TOI",
  "session_id":   null
}
```
Returns:
```json
{
  "session_id": "uuid",
  "results": {
    "no_personalization": { "headline": "...", "latency_ms": 1200 },
    "rag_bm25":           { "headline": "...", "latency_ms": 1800 },
    "stylevector":        { "headline": "...", "latency_ms": 2100 },
    "cold_start_sv":      { "headline": "...", "rouge_l": 0.68, "latency_ms": 2400 }
  }
}
```

### `GET /api/history?user_id=xxx`
Returns last 30 sessions (no message bodies).

### `GET /api/history/<session_id>`
Returns full session with all messages.

### `DELETE /api/history/<session_id>`
Deletes a session.

---

## Connecting Your Modal LLM

When your LLM is deployed, open `backend/routes/generate.py` and update `call_llm()`.

The function currently assumes:
```
POST {MODAL_LLM_URL}/generate
Body: { method, source_text, author_id, publication }
Response: { headline, rouge_l?, latency_ms? }
```

Adjust the URL path, payload keys, and response parsing to match your actual endpoint.

---

## MongoDB Collections

### `journalists`
```json
{ "id": "toi_ps", "name": "Priya Sharma", "publication": "TOI",
  "publication_label": "Times of India", "articles_count": 312 }
```

### `chat_sessions`
```json
{
  "session_id": "uuid",
  "user_id": "google_xxx",
  "author_id": "toi_ps",
  "publication": "TOI",
  "preview": "First 120 chars...",
  "messages": [{ "source_text": "...", "results": {...} }],
  "created_at": 1234567890,
  "updated_at": 1234567890
}
```

---

## Dark Mode

Fully supported via Tailwind `dark:` classes. Toggle persists in `localStorage`. Mirrors your Stitch design exactly — light and dark tokens are identical to `stylevector_flux` design system.
