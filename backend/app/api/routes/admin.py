from fastapi import APIRouter, Header, HTTPException, Depends
from app.core.database import db
from app.models.user import User
from app.models.domain import SkinScan
from app.core.config import settings
from typing import Optional

router = APIRouter()

async def verify_admin(x_admin_password: Optional[str] = Header(None)):
    if not x_admin_password or x_admin_password != settings.ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Admin Password")
    return True

@router.get("/stats", dependencies=[Depends(verify_admin)])
async def get_admin_stats():
    # Count total users
    total_users = await db.engine.count(User)
    
    # Count total scans
    total_scans = await db.engine.count(SkinScan)
    
    # Get recent users (last 10)
    recent_users = await db.engine.find(User, sort=User.created_at.desc(), limit=10)
    
    # Get recent scans (last 10)
    recent_scans = await db.engine.find(SkinScan, sort=SkinScan.created_at.desc(), limit=10)
    
    return {
        "total_users": total_users,
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
                "disease": s.disease_detected,
                "confidence": s.confidence_score,
                "created_at": s.created_at
            } for s in recent_scans
        ]
    }
