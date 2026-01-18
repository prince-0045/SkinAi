from odmantic import Model, Field
from datetime import datetime
from typing import Optional

class User(Model):
    name: str
    email: str = Field(unique=True)
    hashed_password: Optional[str] = None
    auth_provider: str = "email"  # 'email' or 'google'
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    # For Pydantic v2/ODMantic compatibility
    model_config = {
        "collection": "users"
    }
