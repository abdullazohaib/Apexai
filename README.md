# AI Response Comparator & Synthesizer

A production-ready FastAPI web application that scores, compares, and synthesizes responses from ChatGPT, Gemini, and Claude into a single best answer using a rule-based ML scoring engine.

---

## Project Structure

```
ai_comparator/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, routes, lifespan
│   ├── config.py                # Pydantic settings (env-configurable)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py            # /ask, /compare, /dashboard/stats, /history
│   │   └── schemas.py           # Pydantic request/response models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── logger.py            # Rotating file + console logging
│   │   ├── cache.py             # Thread-safe in-memory TTL cache
│   │   ├── scorer.py            # Rule-based 5-dimension scoring engine
│   │   └── synthesizer.py       # Cross-model answer synthesizer
│   ├── db/
│   │   ├── __init__.py
│   │   └── models.py            # SQLAlchemy async models + DB init
│   ├── templates/
│   │   ├── index.html           # Main comparator UI
│   │   └── dashboard.html       # Admin analytics dashboard
│   └── static/                  # (CSS/JS assets if needed)
├── logs/                        # Rotating log files (auto-created)
├── data.db                      # SQLite database (auto-created)
├── run.py                       # Entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
cd ai_comparator
pip install -r requirements.txt
```

### 2. Run the app

```bash
python run.py
```

Or with uvicorn directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open the app

| URL | Description |
|-----|-------------|
| http://localhost:8000 | Main comparator UI |
| http://localhost:8000/dashboard | Admin analytics dashboard |
| http://localhost:8000/docs | Interactive API docs (Swagger UI) |
| http://localhost:8000/redoc | ReDoc API docs |

---

## Configuration

Create a `.env` file in the project root to override defaults:

```env
DEBUG=false
CACHE_TTL=3600
IDEAL_MIN_WORDS=50
IDEAL_MAX_WORDS=400
LOG_LEVEL=INFO

# Scoring weights (must sum to 1.0)
WEIGHT_KEYWORD=0.30
WEIGHT_LENGTH=0.15
WEIGHT_READABILITY=0.20
WEIGHT_COSINE=0.20
WEIGHT_FACTUAL=0.15
```

---

## API Endpoints

### `POST /api/v1/ask`
Multipart form — accepts question, three responses, and optional file upload.

**Form fields:**
- `question` (str) — user's question
- `chatgpt` (str) — ChatGPT's response
- `gemini` (str) — Gemini's response
- `claude` (str) — Claude's response
- `file` (file, optional) — `.txt`, `.md`, `.csv`, `.pdf`

### `POST /api/v1/compare`
JSON endpoint — same functionality as `/ask`.

```json
{
  "question": "What is quantum entanglement?",
  "responses": {
    "chatgpt": "...",
    "gemini": "...",
    "claude": "..."
  }
}
```

### `GET /api/v1/dashboard/stats`
Returns aggregate analytics: total queries, cache stats, winner distribution, avg scores, recent 10 queries.

### `GET /api/v1/history?limit=50`
Returns paginated query history.

### `DELETE /api/v1/cache`
Clears the in-memory cache.

---

## Scoring Engine

Each response is scored on **5 dimensions**:

| Dimension | Method | Weight |
|-----------|--------|--------|
| **Keyword Relevance** | Token overlap between question & response | 30% |
| **Length Score** | Gaussian penalty outside ideal word count range | 15% |
| **Readability** | Flesch Reading Ease normalised to 0–1 | 20% |
| **Cosine Similarity** | TF-IDF cosine sim vs other responses | 20% |
| **Factual Consistency** | Jaccard similarity vs peer responses | 15% |

Weights are fully configurable via environment variables.

---

## Synthesis Strategy

1. Start with the highest-scored response as the base
2. Extract sentences from runner-ups with **novel information** (low overlap with winner)
3. Deduplicate using Jaccard similarity threshold
4. Prepend source attribution header
5. Append "Additional Insights" paragraph from cross-model unique content

---

## Caching

The in-memory `InMemoryCache` (thread-safe, TTL-based) caches results keyed on `SHA256(question + responses)`. To upgrade to Redis, replace the `cache` singleton in `app/core/cache.py` with a Redis client implementing the same `.get()` / `.set()` / `.delete()` / `.stats()` interface.

---

## Logging

All logs are written to `logs/app.log` with rotating file handler (10 MB per file, 5 backups) and also to stdout. Log level is configurable via `LOG_LEVEL`.

---

## Future Extensions

- **Real AI connectors**: Add `app/connectors/openai.py`, `app/connectors/gemini.py` etc. — each implementing `async def complete(question: str) -> str`
- **Redis cache**: Swap `InMemoryCache` for `redis.asyncio.Redis`
- **Authentication**: Add JWT middleware to protect `/dashboard` and `/history`
- **Async scoring**: Run scoring in a `ThreadPoolExecutor` for CPU-bound NLP ops
- **Export**: Add CSV/JSON export endpoint for query history