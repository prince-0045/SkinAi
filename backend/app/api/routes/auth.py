from fastapi import APIRouter, Depends, HTTPException, status, Body
from app.models.user import User
from app.models.domain import OTPLog
from app.core.auth import get_password_hash, verify_password, create_access_token
from app.core.database import get_db
from app.services.email import send_otp_email, send_welcome_email
from app.core.config import settings
from odmantic import AIOEngine
from datetime import datetime, timedelta
import random
import string
from google.oauth2 import id_token
from google.auth.transport import requests

router = APIRouter()

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(
    name: str = Body(...),
    email: str = Body(...),
    password: str = Body(...),
    db: AIOEngine = Depends(get_db)
):
    # Check if user exists
    existing_user = await db.find_one(User, User.email == email)
    if existing_user:
        if existing_user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        else:
            # User exists but failed verification previously. Delete to allow retry.
            await db.delete(existing_user)
    
    try:
        hashed_password = get_password_hash(password)
    except Exception as e:
        print(f"DEBUG: Hashing Failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Hashing Error: {str(e)}")

    # Create new user
    new_user = User(
        name=name,
        email=email,
        hashed_password=hashed_password,
        is_verified=False,
        auth_provider="email"
    )
    await db.save(new_user)
    
    # Generate and send OTP
    otp = generate_otp()
    otp_log = OTPLog(
        email=email,
        otp=otp,
        expires_at=datetime.utcnow() + timedelta(minutes=10)
    )
    await db.save(otp_log)
    
    email_sent = False
    import asyncio
    try:
        # Enforce a short 5-second timeout for email sending in dev
        print(f"DEBUG: sending email to {email}")
        await asyncio.wait_for(send_otp_email(email, otp), timeout=5.0)
        email_sent = True
    except Exception as e:
        print(f"Failed to send email (timeout or error): {e}")
        # Continue to return OTP in debug/fallback mode
    
    response = {"message": "User created. Please verify your email."}
    # Always send OTP in response for testing if email fails
    if not email_sent: 
        response["otp_debug"] = otp
        print(f"DEBUG: Returning OTP in response: {otp}")
        
    return response

@router.post("/login")
async def login(
    email: str = Body(...),
    password: str = Body(...),
    db: AIOEngine = Depends(get_db)
):
    user = await db.find_one(User, User.email == email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    if user.auth_provider == 'google':
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please use Google Sign-In"
        )
    
    if not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
        
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email not verified"
        )
        
    access_token = create_access_token(data={"sub": user.email})
    
    user.last_login = datetime.utcnow()
    await db.save(user)
    
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "user": {
            "name": user.name, 
            "email": user.email, 
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
    }

@router.post("/google")
async def google_login(
    token: str = Body(..., embed=True),
    db: AIOEngine = Depends(get_db)
):
    try:
        # Verify the access token by fetching user info
        import requests
        response = requests.get(
            'https://www.googleapis.com/oauth2/v3/userinfo',
            headers={'Authorization': f'Bearer {token}'}
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Google Access Token"
            )
            
        id_info = response.json()
        
        email = id_info.get('email')
        name = id_info.get('name', '')
        
        if not email:
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Google Account email not found"
            )
        
        # Check if user exists
        user = await db.find_one(User, User.email == email)
        
        if not user:
            # Create new user
            user = User(
                name=name,
                email=email,
                is_verified=True, # Google emails are verified
                auth_provider="google"
            )
            await db.save(user)
            try:
                await send_welcome_email(user.email, user.name)
            except Exception as e:
                print(f"Failed to send welcome email: {e}")
        elif user.auth_provider != 'google':
            # Link account or warn? For now, we'll just allow it but maybe update provider or keep as is.
            # Ideally we should merge or handle gracefully. Let's just log them in.
            pass
            
        access_token = create_access_token(data={"sub": user.email})
        
        user.last_login = datetime.utcnow()
        await db.save(user)
        
        return {
            "access_token": access_token, 
            "token_type": "bearer", 
            "user": {
                "name": user.name, 
                "email": user.email,
                "created_at": user.created_at.isoformat() if user.created_at else None
            }
        }
        
    except Exception as e:
        print(f"Google Login Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Google Login Failed: {str(e)}"
        )

@router.post("/verify-otp")
async def verify_otp(
    email: str = Body(...),
    otp: str = Body(...),
    db: AIOEngine = Depends(get_db)
):
    otp_record = await db.find_one(OTPLog, (OTPLog.email == email) & (OTPLog.otp == otp) & (OTPLog.verified == False))
    
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
            print(f"Failed to send welcome email: {e}")
        
    return {"message": "Email verified successfully"}
