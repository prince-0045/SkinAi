from datetime import datetime, timedelta
from jose import jwt
from app.core.config import settings
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHash

# Initialize Argon2 password hasher
pwd_context = PasswordHasher()


def verify_password(plain_password, hashed_password):
    """Verify a plain password against an argon2 hash."""
    try:
        pwd_context.verify(hashed_password, plain_password)
        return True
    except (VerifyMismatchError, InvalidHash):
        return False


def get_password_hash(password):
    """Hash a password using argon2."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt
