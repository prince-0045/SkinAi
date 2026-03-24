from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from app.core.config import settings
from pydantic import EmailStr
from pathlib import Path

# --- NEW: AWS SES SDK Logic (Bypasses Port Blocking) ---
ses_client = boto3.client(
    'ses',
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
)

# --- OLD: SMTP Configuration (Commented out for reference) ---
# from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
# conf = ConnectionConfig(
#     MAIL_USERNAME=settings.MAIL_USERNAME,
#     MAIL_PASSWORD=settings.MAIL_PASSWORD,
#     MAIL_FROM=settings.MAIL_FROM,
#     MAIL_PORT=settings.MAIL_PORT,
#     MAIL_SERVER=settings.MAIL_SERVER,
#     MAIL_STARTTLS=settings.MAIL_STARTTLS,
#     MAIL_SSL_TLS=settings.MAIL_SSL_TLS,
#     USE_CREDENTIALS=True,
#     VALIDATE_CERTS=False
# )


async def send_otp_email(email: EmailStr, otp: str):
    print(f"Attempting to send OTP email via AWS SES to {email}")
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

    # --- AWS SES Implementation ---
    try:
        response = ses_client.send_email(
            Source=settings.MAIL_FROM,
            Destination={'ToAddresses': [email]},
            Message={
                'Subject': {'Data': 'SkinCare AI - Verification Code'},
                'Body': {
                    'Html': {'Data': html}
                }
            }
        )
        print(f"SUCCESS: Email sent via SES. Message ID: {response['MessageId']}")
    except ClientError as e:
        print(f"SES CRITICAL ERROR: {e.response['Error']['Message']}")
        raise e
    except Exception as e:
        print(f"General Error sending email: {e}")
        raise e

    # --- OLD SMTP IMPLEMENTATION (Commented) ---
    # message = MessageSchema(
    #     subject="SkinCare AI - Verification Code",
    #     recipients=[email],
    #     body=html,
    #     subtype=MessageType.html,
    # )
    # fm = FastMail(conf)
    # try:
    #     print(f"DEBUG: Connection Config: {conf.MAIL_SERVER}:{conf.MAIL_PORT}, User={conf.MAIL_USERNAME}")
    #     await fm.send_message(message)
    #     print("Email sent successfully")
    # except Exception as e:
    #     print(f"Error sending email: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     raise e


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

    # --- AWS SES Implementation ---
    try:
        response = ses_client.send_email(
            Source=settings.MAIL_FROM,
            Destination={'ToAddresses': [email]},
            Message={
                'Subject': {'Data': 'Welcome to SkinCare AI'},
                'Body': {
                    'Html': {'Data': html}
                }
            }
        )
        print(f"SUCCESS: Welcome email sent via SES. Message ID: {response['MessageId']}")
    except Exception as e:
        print(f"Error sending welcome email via SES: {e}")
        # Not raising here to prevent signup flow from breaking if welcome email fails

    # --- OLD SMTP IMPLEMENTATION (Commented) ---
    # message = MessageSchema(...)
    # fm = FastMail(conf)
    # await fm.send_message(message)
