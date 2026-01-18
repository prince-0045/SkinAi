import asyncio
from app.services.email import send_otp_email

async def test_email():
    try:
        print("Sending test email...")
        await send_otp_email("shreypatel0605@gmail.com", "123456")
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_email())
