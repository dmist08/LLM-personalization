import axios from 'axios';

// Base URL: in dev uses Vite proxy → Flask on :5000
// In production this is your Vercel env var pointing to Modal
const BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 300000, // 5min — Modal cold starts can take 90-180s
});

// ─── Warm-up ping ──────────────────────────────────────────────────────────
// Fire-and-forget health check on page load to wake Modal containers.
// This ensures the backend is warm by the time the user interacts.
(async () => {
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      await api.get('/health', { timeout: 30000 });
      console.log('[API] Backend is warm');
      break;
    } catch (err) {
      console.log(`[API] Warm-up attempt ${attempt}/3 failed, retrying...`);
      if (attempt < 3) await new Promise(r => setTimeout(r, attempt * 3000));
    }
  }
})();

// ─── Authors ───────────────────────────────────────────────────────────────
export const getAuthors = async (publication = null) => {
  const params = publication ? { publication } : {};
  const { data } = await api.get('/authors', { params });
  return data; // { authors: [{id, name, publication, articles_count}] }
};

// ─── Generate Headlines ────────────────────────────────────────────────────
// Returns: {
//   session_id,
//   results: {
//     no_personalization: { headline, latency_ms },
//     rag_bm25:           { headline, latency_ms },
//     stylevector:        { headline, latency_ms },
//     cold_start_sv:      { headline, rouge_l, latency_ms }
//   }
// }
export const generateHeadlines = async ({ sourceText, publication, authorId, sessionId = null }) => {
  const { data } = await api.post('/generate', {
    source_text: sourceText,
    publication,
    author_id: authorId,
    session_id: sessionId,
  });
  return data;
};

// ─── Streaming variant (SSE) ───────────────────────────────────────────────
// Use this when your Modal LLM supports streaming
// Returns an EventSource you must close when done
export const generateHeadlinesStream = (payload, onMessage, onDone, onError) => {
  // We POST first to create a job, then open an SSE stream for results
  api.post('/generate/stream', {
    source_text: payload.sourceText,
    publication: payload.publication,
    author_id: payload.authorId,
    session_id: payload.sessionId || null,
  }).then(({ data }) => {
    const { job_id } = data;
    const es = new EventSource(`${BASE_URL}/generate/stream/${job_id}`);
    es.onmessage = (e) => {
      const parsed = JSON.parse(e.data);
      if (parsed.done) {
        es.close();
        onDone(parsed);
      } else {
        onMessage(parsed);
      }
    };
    es.onerror = (err) => {
      es.close();
      onError(err);
    };
    return es;
  }).catch(onError);
};

// ─── Chat History ──────────────────────────────────────────────────────────
export const getChatHistory = async (userId) => {
  const { data } = await api.get('/history', { params: { user_id: userId } });
  return data; // { sessions: [{id, author, publication, preview, created_at}] }
};

export const getChatSession = async (sessionId) => {
  const { data } = await api.get(`/history/${sessionId}`);
  return data; // full session object
};

export const deleteChatSession = async (sessionId) => {
  const { data } = await api.delete(`/history/${sessionId}`);
  return data;
};

export default api;
