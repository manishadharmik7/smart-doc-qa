# Smart Document Q&A System

A production-ready RAG (Retrieval-Augmented Generation) system where users register, upload PDF/TXT documents, and ask questions in natural language. Built with **FastAPI, ChromaDB, Groq (Llama-3.1), sentence-transformers, and Streamlit**.

> Built as a take-home assignment for HyperMindZ AI Engineer role.

---

## Live Demo

- **App URL:** `https://smart-doc-app-a8qp7izlxq6sns8n9dgfr9.streamlit.app/`
- **API Docs:** `https://smart-doc-qa.onrender.com/docs`
- **Test Email:** `test@gmail.com`
- **Test Password:** `test123`

> Note: Backend is on Render free tier - first request may take ~30s if the service was sleeping.

---

## Architecture

```
User uploads PDF/TXT
        |
FastAPI receives file (validates type + size <= 10MB)
        |
PyPDF2 extracts raw text
        |
LangChain RecursiveCharacterTextSplitter
-> chunks: 800 chars, 100 overlap
        |
sentence-transformers (all-MiniLM-L6-v2)
-> 384-dim embedding vectors  [LOCAL, FREE, no API key]
        |
ChromaDB stores vectors + metadata
(user_id, doc_id, filename, chunk_index)
        |
User asks a question
        |
Question -> embedding (same local model)
        |
ChromaDB cosine similarity search
(filtered by user_id -> strict data isolation)
        |
Top-5 chunks retrieved
        |
Groq API -> llama-3.1-8b-instant
-> grounded answer + source citations
        |
Streamlit displays answer + expandable sources
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Backend | **FastAPI** | Async, auto-docs at /docs, production-grade Python |
| Frontend | **Streamlit** | Rapid AI app UI, no JavaScript needed |
| Vector DB | **ChromaDB** | Persistent, metadata filtering, per-user isolation |
| Embeddings | **all-MiniLM-L6-v2** | Free, local, top-10 MTEB leaderboard quality |
| LLM | **Groq - llama-3.1-8b-instant** | 14,400 free requests/day, sub-second latency |
| Auth | **JWT + bcrypt** | Stateless, scalable, secure password hashing |
| Database | **SQLite + SQLAlchemy** | Zero-config, one-line switch to PostgreSQL |
| PDF parsing | **PyPDF2** | Lightweight, handles text-based PDFs |

---

## Quick Start (Local)

### 1. Clone and install
```bash
git clone <repo-url>
cd smart-doc-qa
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create your .env
```bash
copy .env.example .env
# Edit .env: add your Groq key and a random SECRET_KEY
```

Get a free Groq API key at: https://console.groq.com
Generate SECRET_KEY: python -c "import secrets; print(secrets.token_hex(32))"

### 3. Run backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Run frontend (new terminal)
```bash
cd frontend
streamlit run app.py
```

Open: http://localhost:8501 | API docs: http://localhost:8000/docs

---

## Deployment Guide

### Backend on Render

1. Go to render.com -> New Web Service
2. Connect your GitHub repository
3. Settings:
   - Root Directory: backend
   - Build Command: pip install -r ../requirements.txt
   - Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
   - Environment: Python 3
4. Add Environment Variables in Render dashboard (NOT in code):
   - GROQ_API_KEY
   - HUGGINGFACE_API_TOKEN
   - SECRET_KEY
5. Deploy and copy the URL

Note: Render free tier is ephemeral (no persistent disk). Data resets on restart.
For production: upgrade to $7/mo paid plan or use PostgreSQL + Pinecone.

### Frontend on Streamlit Community Cloud

1. Go to share.streamlit.io -> New app
2. Connect GitHub repo
3. Main file: frontend/app.py
4. Add Secret (Advanced settings):
   BACKEND_URL = "https://your-render-url.onrender.com"
5. Deploy and share the URL

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| GROQ_API_KEY | Free Groq API key from console.groq.com | Yes |
| HUGGINGFACE_API_TOKEN | Free HF token (optional) | No |
| SECRET_KEY | Random 32+ char string for JWT signing | Yes |
| DATABASE_URL | SQLite path (default: sqlite:///./smart_doc_qa.db) | No |
| CHROMA_DB_PATH | ChromaDB storage path (default: ./chroma_db) | No |
| BACKEND_URL | Backend URL for Streamlit frontend (cloud deployment only) | Cloud only |

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | /health | No | Health check |
| GET | /docs | No | Interactive Swagger UI |
| POST | /auth/register | No | Create account |
| POST | /auth/login | No | Login, get JWT token |
| GET | /auth/me | Yes | Current user info |
| POST | /documents/upload | Yes | Upload + process PDF/TXT |
| GET | /documents/list | Yes | List your documents |
| DELETE | /documents/{id} | Yes | Delete document |
| POST | /qa/ask | Yes | Ask question, get answer + sources |

---

## Project Structure

```
smart-doc-qa/
|-- backend/
|   |-- main.py          # FastAPI app + all routes
|   |-- auth.py          # JWT auth, bcrypt, get_current_user dependency
|   |-- rag.py           # Full RAG pipeline
|   |-- database.py      # SQLite setup, User + Document models
|   |-- models.py        # Pydantic schemas
|-- frontend/
|   |-- app.py           # Streamlit UI (Login, Documents, Q&A)
|-- docs/
|   |-- architecture.md  # Architecture decisions + trade-offs
|-- .env.example         # Environment variable template
|-- .gitignore
|-- requirements.txt
```

---

## Security

- Passwords hashed with bcrypt (never stored plain text)
- JWT tokens expire after 24 hours
- Every query filtered by user_id - users cannot access each other's data
- API keys in environment variables only (never in code)
- File type validation (PDF/TXT) + 10MB size limit
- Ownership check before delete operations

---

## AI Tool Usage

- **Claude Code (Anthropic)**: Used for boilerplate scaffolding and code suggestions
- All architectural decisions, design choices, and implementations made independently
- AI suggestions were reviewed, modified, or rejected based on project requirements
- Core logic (RAG pipeline, auth flow, data isolation strategy) written and understood personally

---

## Known Limitations and Production Improvements

| Current | Production Improvement |
|---|---|
| SQLite | PostgreSQL on RDS |
| ChromaDB local | Pinecone or pgvector |
| One file at a time | Batch upload + async background tasks |
| Ephemeral storage (free tier) | Persistent disk or cloud DB |
| No rate limiting | Per-user rate limiting |
| Text PDFs only | OCR for scanned PDFs |
| No streaming | LLM token streaming |
