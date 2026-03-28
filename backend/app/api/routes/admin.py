from fastapi import APIRouter, Header, HTTPException, Depends
from app.core.database import db
from app.models.user import User
from app.models.domain import SkinScan
from app.core.config import settings
from typing import Optional
import math

def safe_float(value, default=0.0):
    """Return default if value is None, NaN, or Infinity."""
    try:
        if value is None or not math.isfinite(float(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

router = APIRouter()

async def verify_admin(x_admin_password: Optional[str] = Header(None)):
    if not x_admin_password or x_admin_password != settings.ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Admin Password")
    return True

from datetime import datetime, timedelta

@router.get("/stats", dependencies=[Depends(verify_admin)])
async def get_admin_stats():
    from app.models.domain import ActiveSession
    
    # Count total users
    total_users = await db.engine.count(User)
    
    # Count active users (last 24h)
    twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
    active_users_24h = await db.engine.count(User, User.last_login >= twenty_four_hours_ago)
    
    # Count live users (last 2 minutes)
    two_minutes_ago = datetime.utcnow() - timedelta(minutes=2)
    live_users = await db.engine.count(ActiveSession, ActiveSession.last_seen_at >= two_minutes_ago)
    
    # Count total scans
    total_scans = await db.engine.count(SkinScan)
    
    # Get recent users (last 10)
    recent_users = await db.engine.find(User, sort=User.created_at.desc(), limit=10)
    
    # Get recent scans (last 10)
    recent_scans = await db.engine.find(SkinScan, sort=SkinScan.created_at.desc(), limit=10)
    
    return {
        "total_users": total_users,
        "active_users_24h": active_users_24h,
        "live_users": live_users,
        "total_scans": total_scans,
        "recent_users": [
            {
                "name": u.name,
                "email": u.email,
                "created_at": u.created_at
            } for u in recent_users
        ],
        "recent_scans": [
            {
                "disease": s.disease_detected or "Unknown",
                "confidence": safe_float(s.confidence_score),
                "created_at": s.created_at
            } for s in recent_scans
        ]
    }
