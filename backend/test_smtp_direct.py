import smtplib
from app.core.config import settings

def test_smtp_direct():
    print(f"Testing direct SMTP with settings:")
    print(f"Server: {settings.MAIL_SERVER}")
    print(f"Port: 587")
    print(f"User: {settings.MAIL_USERNAME}")
    
    try:
        print("\nConnecting to server...")
        server = smtplib.SMTP(settings.MAIL_SERVER, 587, timeout=10)
        print("Starting TLS...")
        server.starttls()
        print("Logging in...")
        server.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)
        print("SUCCESS: Login successful!")
        server.quit()
    except Exception as e:
        print(f"\nFAILURE: {e}")

if __name__ == "__main__":
    test_smtp_direct()
