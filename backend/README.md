# Backend — Flask API Server

Flask API server that proxies headline generation requests to the Modal GPU endpoint and manages chat session history via MongoDB.

## Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
python app.py
```

Server starts at `http://localhost:5000`.

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MODAL_LLM_URL` | Modal GPU endpoint URL | `https://dkmist08--stylevector-llm-api.modal.run` |
| `MONGODB_URI` | MongoDB Atlas connection string | `mongodb+srv://...` |
| `MONGODB_DB` | Database name | `stylevector` |
| `PORT` | Server port | `5000` |
| `FLASK_DEBUG` | Debug mode (disable in prod) | `false` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `http://localhost:5173` |
| `LLM_TIMEOUT_SECONDS` | Timeout for Modal calls | `120` |

## API Endpoints

### `GET /api/health`
Health check.
```json
{"status": "ok", "service": "stylevector-backend"}
```

### `GET /api/authors`
List all 42 journalists. Optional `?publication=TOI` filter.
```json
{
  "count": 42,
  "authors": [
    {"id": "alok_chamaria", "name": "Alok Chamaria", "publication": "TOI", "articles_count": 383, "class": "rich"}
  ]
}
```

### `POST /api/generate`
Generate headlines using 4 methods in parallel.

**Request:**
```json
{
  "source_text": "Article body text...",
  "author_id": "alok_chamaria",
  "publication": "TOI"
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "results": {
    "no_personalization": {"headline": "...", "latency_ms": 2500},
    "rag_bm25": {"headline": "...", "latency_ms": 3100},
    "stylevector": {"headline": "...", "latency_ms": 2800},
    "cold_start_sv": {"headline": "...", "latency_ms": 2600}
  },
  "errors": []
}
```

### `GET /api/history`
List chat sessions. `?user_id=anonymous&limit=30`

### `GET /api/history/<session_id>`
Get full session with all messages.

### `DELETE /api/history/<session_id>`
Delete a chat session.

## Architecture

```
Frontend ──▶ Flask API ──▶ Modal GPU (4 parallel calls)
                │
                └──▶ MongoDB Atlas (session persistence)
```

The backend fires 4 concurrent POST requests to the Modal endpoint (one per generation method) and aggregates the results.

## Files

```
backend/
├── app.py                 Flask app factory
├── routes/
│   ├── authors.py         /api/authors endpoint
│   ├── generate.py        /api/generate endpoint
│   └── history.py         /api/history endpoints
├── db/
│   └── mongo.py           MongoDB connection
├── author_metadata.json   42 journalist metadata
├── requirements.txt       Python dependencies
├── .env.example           Environment template
└── seed_db.py             Database seeder
```
