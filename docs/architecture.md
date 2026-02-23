# Architecture Decisions

## Why This Stack?

### FastAPI (Backend)
- Async by default — handles concurrent uploads without blocking
- Auto-generates interactive API docs at /docs (great for demos)
- Pydantic integration = automatic request validation
- Production-grade: used by Netflix, Uber

### Streamlit (Frontend)
- Fastest way to build AI app UIs in Python
- No JavaScript needed — perfect for ML/AI engineers
- session_state simulates React-like state management
- Can deploy instantly on Streamlit Cloud

### ChromaDB (Vector Store)
- Persists data to disk (survives restarts) — unlike FAISS in-memory
- Native metadata filtering — lets us isolate data per user_id
- Simple Python API — no server to manage
- Production upgrade path: swap for Pinecone or Weaviate

### SQLite (User + Document Records)
- Zero configuration
- SQLAlchemy ORM = swap to PostgreSQL by changing one env var
- Sufficient for this scale; in production: use PostgreSQL on RDS

### JWT Authentication
- Stateless — no server-side sessions
- Scales horizontally (any server instance can verify)
- 24-hour expiry — balance between security and UX
- bcrypt hashing — computationally expensive to brute-force

## Database Schema

```
users
  id            INTEGER PRIMARY KEY
  email         TEXT UNIQUE NOT NULL
  hashed_password TEXT NOT NULL
  created_at    DATETIME

documents
  id            INTEGER PRIMARY KEY
  user_id       INTEGER NOT NULL (FK → users.id)
  filename      TEXT NOT NULL
  chroma_collection_id TEXT
  uploaded_at   DATETIME
```

## ChromaDB Metadata Schema

Each chunk stored with:
```json
{
  "user_id": "123",
  "document_id": "45",
  "filename": "report.pdf",
  "chunk_index": 3
}
```

Query filter: `{"user_id": {"$eq": "123"}}` — ensures data isolation.

## Chunking Strategy

- **chunk_size = 800 chars** (~600 words) — enough context for a meaningful answer
- **chunk_overlap = 100 chars** — consecutive chunks share context, nothing lost at boundaries
- **Separator priority**: paragraphs → sentences → words → characters

## Prompt Engineering

System prompt enforces:
1. Answer only from provided context
2. Say "I don't know" if answer not in context
3. Cite the source document
4. Temperature = 0 for deterministic, factual responses

## AI Tool Usage

- **Claude Code (Anthropic)**: Architecture design, code scaffolding
- **Code reviewed and understood** before accepting suggestions
- All architectural decisions made independently

## Trade-offs Made

| Decision | Trade-off |
|---|---|
| Streamlit over React | Faster to build, less customizable UI |
| SQLite over PostgreSQL | Zero config but not production-scale |
| ChromaDB over Pinecone | Free/local but not cloud-native |
| Groq/Llama over GPT-4 | 14,400 free requests/day vs paid OpenAI |

## What I Would Add With More Time

1. **RAGAS evaluation pipeline** — measure retrieval quality (faithfulness, relevance)
2. **Hybrid search** — BM25 keyword + vector similarity combined
3. **Re-ranking layer** — cross-encoder model to re-rank retrieved chunks
4. **Async ingestion** — background task for large documents
5. **PostgreSQL + Pinecone** — production-grade storage
6. **Rate limiting** — prevent API abuse
7. **Streaming responses** — LLM tokens stream to UI in real-time

## Production Considerations

- API keys in environment variables (never in code)
- Input validation at every endpoint
- Ownership verification before delete operations
- File size limits (10MB)
- File type validation (PDF/TXT only)
- Error messages that don't expose internal details
