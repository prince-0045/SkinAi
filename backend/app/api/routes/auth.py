from fastapi import APIRouter, Depends, HTTPException, status, Body
from app.models.user import User
from app.models.domain import OTPLog
from app.core.auth import get_password_hash, verify_password, create_access_token
from app.core.database import get_db
from app.services.email import send_otp_email, send_welcome_email
from app.core.config import settings
from app.core.constants import (
    OTP_EXPIRY_MINUTES, LOGIN_MAX_ATTEMPTS, LOGIN_LOCKOUT_SECONDS,
    PASSWORD_MIN_LENGTH
)
from odmantic import AIOEngine
from datetime import datetime, timedelta
from collections import defaultdict
import random
import string
import re
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Brute force protection (in-memory tracker) ──
_login_attempts = defaultdict(list)


def generate_otp():
    return "".join(random.choices(string.digits, k=6))


def validate_password(password: str):
    """Enforce password strength requirements."""
    if len(password) < PASSWORD_MIN_LENGTH:
        raise HTTPException(400, f"Password must be at least {PASSWORD_MIN_LENGTH} characters")
    if not re.search(r'[A-Z]', password):
        raise HTTPException(400, "Password must contain at least one uppercase letter")
    if not re.search(r'[0-9]', password):
        raise HTTPException(400, "Password must contain at least one number")


@router.get("/test-email")
async def test_email_diag(email: str):
    """Diagnostic endpoint to test email delivery and see the error."""
    try:
        from app.services.email import send_otp_email
        await send_otp_email(email, "123456")
        return {"status": "success", "message": f"Test email sent to {email}"}
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_detail": str(e),
            "traceback": traceback.format_exc()
        }

@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(
    name: str = Body(...),
    email: str = Body(...),
    password: str = Body(...),
    db: AIOEngine = Depends(get_db),
):
    # Validate password strength
    validate_password(password)

    # Check if user exists
    existing_user = await db.find_one(User, User.email == email)
    if existing_user:
        if existing_user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
        else:
            await db.delete(existing_user)

    try:
        hashed_password = get_password_hash(password)
    except Exception as e:
        logger.error(f"Password hashing failed: {e}")
        raise HTTPException(status_code=500, detail="Account creation failed. Please try again.")

    # Generate OTP
    otp = generate_otp()
    otp_log = OTPLog(
        email=email, otp=otp, expires_at=datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)
    )
    await db.save(otp_log)

    # Send OTP email
    email_sent = False
    import asyncio

    try:
        await asyncio.wait_for(send_otp_email(email, otp), timeout=15.0)
        email_sent = True
    except Exception as e:
        logger.warning(f"OTP email failed for {email}: {e}")

    # Create user
    new_user = User(
        name=name,
        email=email,
        hashed_password=hashed_password,
        is_verified=False,
        auth_provider="email",
    )
    await db.save(new_user)

    response = {"message": "User created. Please verify your email."}
    if not email_sent:
        # Never expose OTP in response — log server-side only
        logger.warning(f"OTP delivery failed for {email}, OTP: {otp}")
        response["message"] = "User created. Email delivery may be delayed — please check your inbox or try again."

    return response


@router.post("/login")
async def login(
    email: str = Body(...), password: str = Body(...), db: AIOEngine = Depends(get_db)
):
    # ── Brute force protection ──
    now = time.time()
    recent = [t for t in _login_attempts[email] if now - t < LOGIN_LOCKOUT_SECONDS]
    _login_attempts[email] = recent

    if len(recent) >= LOGIN_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many login attempts. Please try again in {LOGIN_LOCKOUT_SECONDS // 60} minutes."
        )

    user = await db.find_one(User, User.email == email)
    if not user:
        _login_attempts[email].append(now)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    if user.auth_provider == "google":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Please use Google Sign-In"
        )

    if not verify_password(password, user.hashed_password):
        _login_attempts[email].append(now)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Email not verified"
        )

    # Clear failed attempts on success
    _login_attempts.pop(email, None)

    access_token = create_access_token(data={"sub": user.email})

    user.last_login = datetime.utcnow()
    await db.save(user)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "name": user.name,
            "email": user.email,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        },
    }


@router.post("/google")
async def google_login(
    token: str = Body(..., embed=True), db: AIOEngine = Depends(get_db)
):
    try:
        # Use httpx (async) instead of synchronous requests
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {token}"},
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Google Access Token",
            )

        id_info = response.json()

        email = id_info.get("email")
        name = id_info.get("name", "")

        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Google Account email not found",
            )

        # Check if user exists
        user = await db.find_one(User, User.email == email)

        if not user:
            user = User(
                name=name,
                email=email,
                is_verified=True,
                auth_provider="google",
            )
            await db.save(user)
            try:
                await send_welcome_email(user.email, user.name)
            except Exception as e:
                logger.warning(f"Welcome email failed: {e}")
        elif user.auth_provider != "google":
            pass  # Allow cross-provider login

        access_token = create_access_token(data={"sub": user.email})

        user.last_login = datetime.utcnow()
        await db.save(user)

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "name": user.name,
                "email": user.email,
                "created_at": user.created_at.isoformat() if user.created_at else None,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Google Login Failed: {str(e)}",
        )


@router.post("/verify-otp")
async def verify_otp(
    email: str = Body(...), otp: str = Body(...), db: AIOEngine = Depends(get_db)
):
    otp_record = await db.find_one(
        OTPLog,
        (OTPLog.email == email) & (OTPLog.otp == otp) & (OTPLog.verified == False),
    )

    if not otp_record:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    if datetime.utcnow() > otp_record.expires_at:
        raise HTTPException(status_code=400, detail="OTP expired")

    otp_record.verified = True
    await db.save(otp_record)

    user = await db.find_one(User, User.email == email)
    if user:
        user.is_verified = True
        await db.save(user)
        try:
            await send_welcome_email(user.email, user.name)
        except Exception as e:
            logger.warning(f"Welcome email failed: {e}")

    return {"message": "Email verified successfully"}


# ── Forgot Password Flow ──
@router.post("/forgot-password")
async def forgot_password(
    email: str = Body(..., embed=True), db: AIOEngine = Depends(get_db)
):
    """Send a password reset OTP. Always returns success to prevent email enumeration."""
    user = await db.find_one(User, User.email == email)
    if not user or user.auth_provider == "google":
        # Don't reveal whether the account exists
        return {"message": "If an account exists with that email, a reset code has been sent."}

    otp = generate_otp()
    otp_log = OTPLog(
        email=email, otp=otp, expires_at=datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)
    )
    await db.save(otp_log)

    try:
        await send_otp_email(email, otp)
    except Exception as e:
        logger.warning(f"Password reset email failed for {email}: {e}")

    return {"message": "If an account exists with that email, a reset code has been sent."}


@router.post("/reset-password")
async def reset_password(
    email: str = Body(...),
    otp: str = Body(...),
    new_password: str = Body(...),
    db: AIOEngine = Depends(get_db)
):
    """Reset password using OTP verification."""
    validate_password(new_password)

    otp_record = await db.find_one(
        OTPLog,
        (OTPLog.email == email) & (OTPLog.otp == otp) & (OTPLog.verified == False),
    )

    if not otp_record:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    if datetime.utcnow() > otp_record.expires_at:
        raise HTTPException(status_code=400, detail="OTP expired")

    user = await db.find_one(User, User.email == email)
    if not user:
        raise HTTPException(status_code=400, detail="Account not found")

    user.hashed_password = get_password_hash(new_password)
    await db.save(user)

    otp_record.verified = True
    await db.save(otp_record)

    return {"message": "Password reset successful. You can now log in with your new password."}
