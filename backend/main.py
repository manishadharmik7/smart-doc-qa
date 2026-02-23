"""
main.py — FastAPI application: all routes wired together

ROUTES:
  POST /auth/register   → create new user
  POST /auth/login      → login, get JWT token
  POST /documents/upload → upload + process PDF/TXT
  GET  /documents/list  → list user's documents
  DELETE /documents/{id} → delete document + its chunks
  POST /qa/ask          → ask question, get answer + sources
  GET  /health          → health check (for deployment verification)

DESIGN PATTERN:
  - Routes are thin — they validate input, call service functions, return responses
  - Business logic lives in auth.py and rag.py
  - get_current_user dependency protects all document/qa routes
"""

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import get_db, create_tables, Document, User
from models import (
    UserRegister, UserLogin, TokenResponse, UserResponse,
    DocumentResponse, QuestionRequest, AnswerResponse, SourceChunk
)
from auth import get_current_user, register_user, login_user
from rag import ingest_document, query_documents, delete_document_chunks

# --- App Setup ---
app = FastAPI(
    title="Smart Document Q&A API",
    description="RAG-based document Q&A system with user authentication",
    version="1.0.0"
)

# CORS — allows Streamlit frontend (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: restrict to specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create DB tables on startup
@app.on_event("startup")
def startup():
    create_tables()


# --- Health Check ---

@app.get("/health")
def health_check():
    """Used by deployment platform to verify app is running"""
    return {"status": "healthy", "service": "Smart Document Q&A API"}


# --- Auth Routes ---

@app.post("/auth/register", response_model=UserResponse, status_code=201)
def register(payload: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user.
    - Validates email format (Pydantic EmailStr)
    - Hashes password with bcrypt
    - Returns user info (never the password)
    """
    user = register_user(payload.email, payload.password, db)
    return user


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    """
    Login and receive JWT token.
    - Verifies email + password
    - Returns JWT token (valid 24 hours)
    - Frontend stores this token and sends it in Authorization header
    """
    token = login_user(payload.email, payload.password, db)
    return {"access_token": token, "token_type": "bearer"}


@app.get("/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Return current user's info — useful for frontend to show user email"""
    return current_user


# --- Document Routes ---

@app.post("/documents/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload a PDF or TXT document.
    Pipeline: Read bytes → extract text → chunk → embed → store in ChromaDB
    Also saves document record in SQLite for listing/deletion.
    """
    # Validate file type
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF and TXT files are supported"
        )

    # Validate file size (10MB limit)
    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size must be under 10MB"
        )

    # Save document record to SQLite first (to get document_id)
    doc = Document(
        user_id=current_user.id,
        filename=file.filename,
        chroma_collection_id="documents"
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Run ingestion pipeline
    try:
        chunk_count = ingest_document(
            user_id=current_user.id,
            document_id=doc.id,
            filename=file.filename,
            file_bytes=file_bytes
        )
    except ValueError as e:
        # If ingestion fails, remove the DB record too
        db.delete(doc)
        db.commit()
        raise HTTPException(status_code=400, detail=str(e))

    return doc


@app.get("/documents/list", response_model=list[DocumentResponse])
def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all documents uploaded by the current user.
    Filtered by user_id — users can only see their own documents.
    """
    docs = db.query(Document).filter(Document.user_id == current_user.id).all()
    return docs


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a document and all its stored chunks.
    Verifies ownership — users cannot delete other users' documents.
    """
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id  # ownership check
    ).first()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Delete from ChromaDB first, then SQLite
    delete_document_chunks(current_user.id, document_id)
    db.delete(doc)
    db.commit()


# --- Q&A Route ---

@app.post("/qa/ask", response_model=AnswerResponse)
def ask_question(
    payload: QuestionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Ask a question about uploaded documents.
    - Embeds the question
    - Retrieves top-5 relevant chunks from user's documents
    - Sends chunks + question to LLM
    - Returns grounded answer + source citations
    """
    if not payload.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # Count user's documents
    doc_count = db.query(Document).filter(
        Document.user_id == current_user.id
    ).count()

    if doc_count == 0:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one document before asking questions"
        )

    answer, sources = query_documents(
        user_id=current_user.id,
        question=payload.question,
        document_id=payload.document_id
    )

    source_chunks = [
        SourceChunk(
            content=s["content"],
            filename=s["filename"],
            chunk_index=s["chunk_index"]
        )
        for s in sources
    ]

    return AnswerResponse(
        answer=answer,
        sources=source_chunks,
        documents_searched=doc_count
    )
