import resend
from app.core.config import settings
from pydantic import EmailStr
import logging

logger = logging.getLogger(__name__)

# --- Resend API Configuration (Bypasses VPS Port Blocking) ---
if settings.RESEND_API_KEY:
    resend.api_key = settings.RESEND_API_KEY

async def send_otp_email(email: EmailStr, otp: str):
    logger.info(f"Attempting to send OTP email via Resend to {email}")
    html = f"""
    <html>
        <body>
            <div style="font-family: Arial, sans-serif; padding: 20px;">
                <h2>Verify your DermAura Account</h2>
                <p>Your verification code is:</p>
                <h1 style="color: #4A90E2;">{otp}</h1>
                <p>This code expires in 10 minutes.</p>
            </div>
        </body>
    </html>
    """

    try:
        if not settings.RESEND_API_KEY:
            raise ValueError("RESEND_API_KEY is not set in environment")

        params = {
            "from": f"DermAura <{settings.MAIL_FROM}>",
            "to": [email],
            "subject": "DermAura - Verification Code",
            "html": html,
        }
        
        response = resend.Emails.send(params)
        logger.info(f"SUCCESS: Email sent via Resend. ID: {response.get('id')}")
    except Exception as e:
        logger.error(f"Resend Delivery Error: {e}")
        raise e

async def send_welcome_email(email: EmailStr, name: str):
    html = f"""
    <html>
        <body>
            <div style="font-family: Arial, sans-serif; padding: 20px;">
                <h2>Welcome to DermAura!</h2>
                <p>Hi {name},</p>
                <p>Thank you for joining our community. We are excited to help you on your journey to better skin health.</p>
            </div>
        </body>
    </html>
    """
    
    try:
        if not settings.RESEND_API_KEY:
            return

        params = {
            "from": f"DermAura <{settings.MAIL_FROM}>",
            "to": [email],
            "subject": "Welcome to DermAura",
            "html": html,
        }
        resend.Emails.send(params)
        logger.info("SUCCESS: Welcome email sent via Resend")
    except Exception as e:
        logger.error(f"Error sending welcome email via Resend: {e}")
