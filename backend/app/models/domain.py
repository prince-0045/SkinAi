from odmantic import Model, Field
from datetime import datetime
from typing import Optional, List

class SkinScan(Model):
    user_id: str
    image_url: str
    disease_detected: str
    confidence_score: float
    severity_level: str
    description: Optional[str] = None
    recommendation: Optional[str] = None
    do_list: Optional[List[str]] = None
    dont_list: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "collection": "skin_scans"
    }



class ActiveSession(Model):
    user_id: str
    last_seen_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "collection": "active_sessions"
    }

class OTPLog(Model):
    email: str
    otp: str
    expires_at: datetime
    verified: bool = False
    
    model_config = {
        "collection": "otp_logs"
    }
