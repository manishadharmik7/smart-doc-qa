"""
auth.py — Authentication: Register, Login, JWT tokens, Password hashing

HOW IT WORKS (explain to interviewer):

1. REGISTER:
   - User sends email + password
   - We hash the password with bcrypt (one-way hash — we can NEVER recover original)
   - Store hashed password in SQLite
   - Never store plain text passwords

2. LOGIN:
   - User sends email + password
   - We fetch user from DB, use bcrypt.verify() to compare
   - If match → generate JWT token
   - JWT = JSON Web Token — a signed string that encodes user_id
   - Frontend stores this token and sends it with every request

3. PROTECTED ROUTES:
   - Every API call sends: Authorization: Bearer <token>
   - We decode the JWT to get user_id
   - This tells us WHO is making the request without hitting DB every time

WHY JWT over sessions?
- Stateless — no server-side session storage needed
- Scales horizontally (any server can verify the token)
- Standard for REST APIs
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv

from database import get_db, User

load_dotenv()

# --- Configuration ---
SECRET_KEY = os.getenv("SECRET_KEY", "fallback-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# bcrypt context for password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer scheme — expects "Authorization: Bearer <token>"
security = HTTPBearer()


# --- Password Utilities ---

def hash_password(password: str) -> str:
    """Hash a plain password using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against stored hash"""
    return pwd_context.verify(plain_password, hashed_password)


# --- JWT Utilities ---

def create_access_token(user_id: int) -> str:
    """
    Create a JWT token encoding the user's ID.
    Token expires after ACCESS_TOKEN_EXPIRE_HOURS.
    """
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),   # 'sub' = subject (standard JWT claim)
        "exp": expire
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[int]:
    """
    Decode a JWT token and return the user_id.
    Returns None if token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
        return user_id
    except (JWTError, TypeError, ValueError):
        return None


# --- FastAPI Dependency: Get Current User ---

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    FastAPI dependency — extracts and validates JWT from Authorization header.
    Used in all protected routes with: current_user: User = Depends(get_current_user)

    If token is missing, invalid, or expired → returns 401 Unauthorized
    """
    token = credentials.credentials
    user_id = decode_token(token)

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user


# --- Auth Service Functions ---

def register_user(email: str, password: str, db: Session) -> User:
    """
    Register a new user.
    Raises HTTPException if email already exists.
    """
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    user = User(
        email=email,
        hashed_password=hash_password(password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def login_user(email: str, password: str, db: Session) -> str:
    """
    Authenticate user and return JWT token.
    Raises HTTPException if credentials are invalid.
    """
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    return create_access_token(user.id)
