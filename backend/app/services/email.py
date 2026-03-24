import boto3
from botocore.exceptions import ClientError
from app.core.config import settings
from pydantic import EmailStr
import logging

logger = logging.getLogger(__name__)

# --- AWS SES SDK Logic (Bypasses VPS Port Blocking) ---
ses_client = boto3.client(
    'ses',
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
)

async def send_otp_email(email: EmailStr, otp: str):
    logger.info(f"Attempting to send OTP email via AWS SES to {email}")
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
        response = ses_client.send_email(
            Source=f"DermAura <{settings.MAIL_FROM}>",
            Destination={'ToAddresses': [email]},
            Message={
                'Subject': {'Data': 'DermAura - Verification Code'},
                'Body': {
                    'Html': {'Data': html}
                }
            }
        )
        logger.info(f"SUCCESS: Email sent via SES. Message ID: {response['MessageId']}")
    except ClientError as e:
        logger.error(f"SES ERROR: {e.response['Error']['Message']}")
        raise e
    except Exception as e:
        logger.error(f"General Error sending email: {e}")
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
        ses_client.send_email(
            Source=f"DermAura <{settings.MAIL_FROM}>",
            Destination={'ToAddresses': [email]},
            Message={
                'Subject': {'Data': 'Welcome to DermAura'},
                'Body': {
                    'Html': {'Data': html}
                }
            }
        )
        logger.info("SUCCESS: Welcome email sent via SES")
    except Exception as e:
        logger.error(f"Error sending welcome email via SES: {e}")
