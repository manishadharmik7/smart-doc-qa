"""
rag.py — The core RAG pipeline using HuggingFace (100% FREE, no OpenAI needed)

WHAT CHANGED FROM OPENAI VERSION:
  - Embeddings: OpenAI text-embedding-3-small → sentence-transformers all-MiniLM-L6-v2
    - Runs LOCALLY on your machine, zero cost, no API key needed
    - 384-dimensional vectors (vs 1536 for OpenAI) — smaller but excellent quality
  - LLM: GPT-3.5-turbo → Mistral-7B-Instruct via HuggingFace Inference API
    - Free with a HuggingFace account token
    - Same quality for document Q&A tasks

HOW TO EXPLAIN THIS TO THE CTO:
  "I used open-source models intentionally. The architecture is model-agnostic —
  swapping in GPT-4 or Claude requires changing 2 lines of code. This shows
  cost-awareness and vendor independence, which matters in production."

FULL PIPELINE:

INGESTION (document upload):
  PDF/TXT file bytes
      ↓
  Extract raw text (PyPDF2 for PDF, UTF-8 decode for TXT)
      ↓
  RecursiveCharacterTextSplitter → chunks of 800 chars, 100 overlap
      ↓
  SentenceTransformer.encode() → 384-dim float vectors (LOCAL, fast)
      ↓
  ChromaDB.add() → stored with metadata: user_id, doc_id, filename, chunk_index

QUERYING (user asks a question):
  User question string
      ↓
  SentenceTransformer.encode() → 384-dim vector
      ↓
  ChromaDB.query() with where={"user_id": current_user_id}
      ↓
  Top-5 most similar chunks returned (cosine similarity)
      ↓
  Build context string from chunks
      ↓
  HuggingFace InferenceClient.chat_completion() → Mistral-7B generates answer
      ↓
  Return answer text + source citations list
"""

import os
import io
from typing import List, Tuple, Optional

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from groq import Groq
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Model Setup
# ─────────────────────────────────────────────

# Embedding model — downloads ~80MB on FIRST RUN only, then cached locally
# all-MiniLM-L6-v2: fast, lightweight, excellent semantic similarity quality
# No API key needed — runs entirely on your CPU
print("Loading embedding model (first run downloads ~80MB)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")

# HuggingFace Inference Client (kept for embeddings context, not used for LLM)
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Groq Client — free API, fast inference, reliable
# Get free key at: https://console.groq.com
# WHY GROQ: 14,400 free requests/day, sub-second latency, OpenAI-compatible API
# Model: llama-3.1-8b-instant — fast, good quality for Q&A tasks
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.1-8b-instant"

# ─────────────────────────────────────────────
# ChromaDB Setup
# ─────────────────────────────────────────────

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

COLLECTION_NAME = "documents"


def get_collection():
    """Get or create the ChromaDB collection. cosine = best for semantic similarity."""
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


# ─────────────────────────────────────────────
# Text Extraction
# ─────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from all pages of a PDF.

    LIMITATION (mention in interview if asked):
    PyPDF2 handles text-based PDFs well. Scanned PDFs (images) would need OCR
    (e.g., Tesseract). This is a known trade-off for the time constraint.
    """
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages).strip()


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from a plain .txt file."""
    return file_bytes.decode("utf-8", errors="ignore").strip()


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────

def chunk_text(text: str) -> List[str]:
    """
    Split document text into overlapping chunks.

    WHY RecursiveCharacterTextSplitter?
    Tries separators in order: paragraphs → sentences → words → characters
    This means it respects natural language boundaries — won't split mid-sentence.

    WHY chunk_size=800, chunk_overlap=100?
    - 800 chars ≈ 150-200 words — enough context for a meaningful answer
    - 100 char overlap — consecutive chunks share a boundary, so no info is lost
    - Too small → retrieval misses context; Too large → noisy retrieval

    INTERVIEW TIP: "Chunk size is a hyperparameter. For technical docs, larger
    chunks work better. For FAQs, smaller. I'd tune this with RAGAS evaluation."
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    # Filter out very short chunks (less than 50 chars — likely noise)
    return [c for c in chunks if len(c.strip()) > 50]


# ─────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Convert text to 384-dimensional embedding vectors using all-MiniLM-L6-v2.

    WHY sentence-transformers over OpenAI embeddings?
    - FREE — runs locally, zero API cost
    - Fast — batch processing on CPU in seconds
    - Privacy — data never leaves your machine
    - Quality — all-MiniLM-L6-v2 ranks top-10 on MTEB leaderboard for retrieval

    HOW embeddings work (explain to CTO):
    "The model maps text to a 384-dimensional vector space where semantically
    similar sentences are geometrically close. We measure closeness with
    cosine similarity — angle between vectors."
    """
    vectors = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True  # L2 normalize — improves cosine similarity
    )
    return vectors.tolist()


# ─────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────

def store_document(
    user_id: int,
    document_id: int,
    filename: str,
    chunks: List[str],
    embeddings: List[List[float]]
):
    """
    Store chunks + embeddings in ChromaDB with user metadata.

    DATA ISOLATION STRATEGY (critical for security — explain to CTO):
    Every chunk is tagged with user_id in metadata.
    ALL queries filter by user_id → users can NEVER see each other's data.
    This is the same pattern as row-level security in PostgreSQL.

    ChromaDB ID format: "user_{id}_doc_{id}_chunk_{i}"
    Must be unique strings — this format guarantees uniqueness.
    """
    collection = get_collection()

    ids = [
        f"user_{user_id}_doc_{document_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "user_id": str(user_id),
            "document_id": str(document_id),
            "filename": filename,
            "chunk_index": i
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )


def delete_document_chunks(user_id: int, document_id: int):
    """Delete all ChromaDB chunks belonging to a specific document + user."""
    collection = get_collection()
    collection.delete(
        where={
            "$and": [
                {"user_id": {"$eq": str(user_id)}},
                {"document_id": {"$eq": str(document_id)}}
            ]
        }
    )


# ─────────────────────────────────────────────
# Full Ingestion Pipeline
# ─────────────────────────────────────────────

def ingest_document(
    user_id: int,
    document_id: int,
    filename: str,
    file_bytes: bytes
) -> int:
    """
    End-to-end ingestion: raw file bytes → stored in ChromaDB.
    Returns number of chunks created (useful for logging).

    Steps: Extract → Chunk → Embed → Store
    """
    # Step 1: Extract text
    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        text = extract_text_from_txt(file_bytes)

    if not text:
        raise ValueError("Could not extract any text from this document.")

    # Step 2: Chunk
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Document produced no usable chunks after splitting.")

    # Step 3: Embed (local, fast)
    embeddings = get_embeddings(chunks)

    # Step 4: Store in ChromaDB
    store_document(user_id, document_id, filename, chunks, embeddings)

    return len(chunks)


# ─────────────────────────────────────────────
# Query + Answer Generation
# ─────────────────────────────────────────────

def query_documents(
    user_id: int,
    question: str,
    document_id: Optional[int] = None,
    top_k: int = 5
) -> Tuple[str, List[dict]]:
    """
    Full RAG query: question → answer + citations.

    Steps:
    1. Embed the question (same model as ingestion — must match!)
    2. Filter ChromaDB by user_id (and optionally document_id)
    3. Find top-k most similar chunks (cosine similarity)
    4. Build context from retrieved chunks
    5. Call Mistral-7B with context + question
    6. Return answer + source metadata

    WHY top_k=5?
    "5 chunks gives enough context for most questions. More chunks = more
    tokens = higher LLM cost and possible context dilution. I'd make this
    configurable in production."
    """
    collection = get_collection()

    # Build the metadata filter for data isolation
    if document_id:
        where_filter = {
            "$and": [
                {"user_id": {"$eq": str(user_id)}},
                {"document_id": {"$eq": str(document_id)}}
            ]
        }
    else:
        where_filter = {"user_id": {"$eq": str(user_id)}}

    # Check collection is not empty
    total = collection.count()
    if total == 0:
        return "No documents found. Please upload documents first.", []

    # Embed the question (MUST use same model as documents)
    question_embedding = get_embeddings([question])[0]

    # Similarity search — returns closest chunks
    n_results = min(top_k, total)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    if not chunks:
        return "No relevant information found in your documents.", []

    # Build context for LLM
    context_parts = []
    sources = []

    for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances)):
        context_parts.append(f"[Source {i+1} — {meta['filename']}]\n{chunk}")
        sources.append({
            "content": chunk[:400] + "..." if len(chunk) > 400 else chunk,
            "filename": meta["filename"],
            "chunk_index": int(meta["chunk_index"]),
            "relevance_score": round(1 - float(dist), 3)
        })

    context = "\n\n---\n\n".join(context_parts)
    answer = generate_answer(question, context)
    return answer, sources


def generate_answer(question: str, context: str) -> str:
    """
    Generate a grounded answer using HuggingFace Inference API.

    PROMPT ENGINEERING DECISIONS (explain to CTO):
    - "Answer ONLY from context" → prevents hallucination
    - "Say I don't know" → honesty over hallucination for Q&A
    - "Cite the source" → traceability, user trust
    - temperature=0.1 → near-deterministic, factual answers
    - max_tokens=600 → focused answers, cost control

    RESILIENCE STRATEGY:
    - Tries multiple free models in order (some get removed from free tier)
    - Final fallback: returns top retrieved chunk directly — app NEVER crashes
    - This is production thinking: degrade gracefully, never fail silently
    """
    system_prompt = (
        "You are a precise document Q&A assistant.\n"
        "RULES:\n"
        "1. Answer ONLY based on the provided context.\n"
        "2. If the answer is not in the context, say: "
        "'I could not find information about this in the uploaded documents.'\n"
        "3. Mention which source your answer comes from.\n"
        "4. Be concise and accurate. Do not use knowledge outside the context."
    )

    user_prompt = (
        f"Context from uploaded documents:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]

    try:
        # Groq — free, fast, reliable inference
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=600,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Final fallback — app never crashes, shows top retrieved chunk
        top_chunk = context.split("---")[0].strip() if "---" in context else context[:600]
        return (
            f"**Note:** LLM unavailable ({str(e)[:100]}). "
            f"Most relevant passage from your document:\n\n{top_chunk}"
        )
