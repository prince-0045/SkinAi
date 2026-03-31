import asyncio
from app.services.email import send_otp_email
from app.core.config import settings

async def test_email():
    print(f"Testing email with settings:")
    print(f"Server: {settings.MAIL_SERVER}")
    print(f"Port: {settings.MAIL_PORT}")
    print(f"User: {settings.MAIL_USERNAME}")
    print(f"From: {settings.MAIL_FROM}")
    
    test_receiver = settings.MAIL_USERNAME # Send to self for testing
    test_otp = "123456"
    
    print(f"\nSending test OTP to {test_receiver}...")
    try:
        await send_otp_email(test_receiver, test_otp)
        print("\nSUCCESS: Email sent!")
    except Exception as e:
        print(f"\nFAILURE: {e}")

if __name__ == "__main__":
    asyncio.run(test_email())
