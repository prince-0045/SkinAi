from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from app.core.config import settings
from pydantic import EmailStr
from pathlib import Path

conf = ConnectionConfig(
    MAIL_USERNAME=settings.MAIL_USERNAME,
    MAIL_PASSWORD=settings.MAIL_PASSWORD,
    MAIL_FROM=settings.MAIL_FROM,
    MAIL_PORT=settings.MAIL_PORT,
    MAIL_SERVER=settings.MAIL_SERVER,
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True,
)


async def send_otp_email(email: EmailStr, otp: str):
    print(f"Attempting to send OTP email to {email}")
    html = f"""
    <html>
        <body>
            <div style="font-family: Arial, sans-serif; padding: 20px;">
                <h2>Verify your SkinCare AI Account</h2>
                <p>Your verification code is:</p>
                <h1 style="color: #4A90E2;">{otp}</h1>
                <p>This code expires in 10 minutes.</p>
            </div>
        </body>
    </html>
    """

    message = MessageSchema(
        subject="SkinCare AI - Verification Code",
        recipients=[email],
        body=html,
        subtype=MessageType.html,
    )

    fm = FastMail(conf)
    try:
        await fm.send_message(message)
        print("Email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")
        raise e


async def send_welcome_email(email: EmailStr, name: str):
    html = f"""
    <html>
        <body>
            <div style="font-family: Arial, sans-serif; padding: 20px;">
                <h2>Welcome to SkinCare AI!</h2>
                <p>Hi {name},</p>
                <p>Thank you for joining our community. We are excited to help you on your journey to better skin health.</p>
            </div>
        </body>
    </html>
    """

    message = MessageSchema(
        subject="Welcome to SkinCare AI",
        recipients=[email],
        body=html,
        subtype=MessageType.html,
    )

    fm = FastMail(conf)
    try:
        await fm.send_message(message)
        print("Welcome email sent successfully")
    except Exception as e:
        print(f"Error sending welcome email: {e}")
        raise e
