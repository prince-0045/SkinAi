from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Body
from app.models.user import User
from app.models.domain import HealingProgress
from app.api.deps import get_current_user
from app.core.database import get_db
from odmantic import AIOEngine
import cloudinary.uploader
from app.core import cloudinary_config
from datetime import datetime
import random

router = APIRouter()

# UPLOAD_DIR = Path("uploads/healing")
# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/record")
async def record_progress(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AIOEngine = Depends(get_db)
):
    try:
        # Save before image
        before_result = cloudinary.uploader.upload(before_image.file, folder="skinai/healing")
        before_url = before_result["secure_url"]
            
        # Save after image
        after_result = cloudinary.uploader.upload(after_image.file, folder="skinai/healing")
        after_url = after_result["secure_url"]
            
        # Calculate improvement (Mock logic)
        improvement = random.uniform(5.0, 30.0) # Mock improvement percentage
        
        progress = HealingProgress(
            user_id=str(current_user.id),
            before_image=before_url,
            after_image=after_url,
            improvement_percentage=improvement
        )
        await db.save(progress)
        
        return progress
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_healing_history(
    current_user: User = Depends(get_current_user),
    db: AIOEngine = Depends(get_db)
):
    history = await db.find(HealingProgress, HealingProgress.user_id == str(current_user.id), sort=HealingProgress.comparison_date.desc())
    return history
