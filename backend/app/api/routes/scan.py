from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.models.user import User
from app.models.domain import SkinScan
from app.api.deps import get_current_user
from app.core.database import get_db
from odmantic import AIOEngine
import shutil
from pathlib import Path
import random
from datetime import datetime

router = APIRouter()

UPLOAD_DIR = Path("uploads/scans")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mock ML function
def predict_skin_disease(image_path):
    diseases = ["Eczema", "Psoriasis", "Melanoma", "Acne", "Benign Keratosis"]
    detected = random.choice(diseases)
    confidence = random.uniform(0.75, 0.99)
    severity = random.choice(["Mild", "Moderate", "Severe"])
    return detected, confidence, severity

@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AIOEngine = Depends(get_db)
):
    try:
        # Save file
        file_path = UPLOAD_DIR / f"{datetime.utcnow().timestamp()}_{file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run inference
        disease, confidence, severity = predict_skin_disease(file_path)
        
        # Save result
        scan = SkinScan(
            user_id=str(current_user.id),
            image_url=str(file_path),
            disease_detected=disease,
            confidence_score=confidence,
            severity_level=severity
        )
        await db.save(scan)
        
        return scan
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_scan_history(
    current_user: User = Depends(get_current_user),
    db: AIOEngine = Depends(get_db)
):
    scans = await db.find(SkinScan, SkinScan.user_id == str(current_user.id), sort=SkinScan.created_at.desc())
    return scans
