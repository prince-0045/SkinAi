from datetime import datetime, timedelta
from jose import jwt
from app.core.config import settings


from passlib.context import CryptContext

# Use argon2 as it's more stable with current Python environments than bcrypt 4.0+
pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password with fallback to plain comparison for legacy data."""
    try:
        # Try to verify with passlib (supports argon2 and bcrypt)
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        # Fallback for plain text passwords stored during bypass mode
        print(f"DEBUG: verify_password fallback check for legacy user: {e}")
        return plain_password == hashed_password

def get_password_hash(password: str) -> str:
    """Hash password using argon2."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt
