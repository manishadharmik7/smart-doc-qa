"""
database.py — SQLite database setup using SQLAlchemy

WHY SQLite?
- Zero configuration, no external DB server needed
- Perfect for this scale (single-user demo / small team)
- In production, swap DATABASE_URL to PostgreSQL — zero code change needed

WHY SQLAlchemy?
- ORM lets us write Python classes instead of raw SQL
- Easy to migrate to PostgreSQL later
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./smart_doc_qa.db")

# connect_args only needed for SQLite (handles threading)
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# --- Database Models (Tables) ---

class User(Base):
    """Users table — stores registered users"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    """Documents table — tracks uploaded documents per user"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    filename = Column(String, nullable=False)
    chroma_collection_id = Column(String, nullable=False)  # reference to ChromaDB
    uploaded_at = Column(DateTime, default=datetime.utcnow)


def create_tables():
    """Create all tables on startup"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency: provides a DB session per request, always closes after"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
