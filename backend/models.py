"""
models.py — Pydantic schemas for request/response validation

WHY Pydantic?
- FastAPI uses Pydantic to validate incoming request data automatically
- If a user sends wrong data type, FastAPI returns a clear error (not a crash)
- Also auto-generates API documentation at /docs

Interview explanation:
"Pydantic models are the contract between frontend and backend.
They define exactly what data shape is expected and returned."
"""

from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List


# --- Auth Schemas ---

class UserRegister(BaseModel):
    """What frontend sends when registering"""
    email: EmailStr          # validates it's a proper email format
    password: str


class UserLogin(BaseModel):
    """What frontend sends when logging in"""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """What backend returns after successful login"""
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """Safe user info to return (never return hashed_password!)"""
    id: int
    email: str
    created_at: datetime

    class Config:
        from_attributes = True  # allows reading from SQLAlchemy model


# --- Document Schemas ---

class DocumentResponse(BaseModel):
    """Document info returned to frontend"""
    id: int
    filename: str
    uploaded_at: datetime

    class Config:
        from_attributes = True


# --- Q&A Schemas ---

class QuestionRequest(BaseModel):
    """What frontend sends when asking a question"""
    question: str
    document_id: Optional[int] = None  # None = search ALL user docs


class SourceChunk(BaseModel):
    """A single retrieved chunk shown as citation"""
    content: str
    filename: str
    chunk_index: int


class AnswerResponse(BaseModel):
    """Full answer with citations"""
    answer: str
    sources: List[SourceChunk]
    documents_searched: int
