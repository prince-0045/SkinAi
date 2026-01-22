from odmantic import Model, Field
from datetime import datetime
from typing import Optional

class SkinScan(Model):
    user_id: str
    image_url: str
    disease_detected: str
    confidence_score: float
    severity_level: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "collection": "skin_scans"
    }



class OTPLog(Model):
    email: str
    otp: str
    expires_at: datetime
    verified: bool = False
    
    model_config = {
        "collection": "otp_logs"
    }
