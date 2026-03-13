from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.models.user import User
from app.models.domain import SkinScan
from app.api.deps import get_current_user
from app.core.database import get_db
from odmantic import AIOEngine
import cloudinary.uploader
from app.core import cloudinary_config
from app.services.ml_model import predict as ml_predict
from datetime import datetime

router = APIRouter()


@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AIOEngine = Depends(get_db)
):
    try:
        # Read the uploaded image bytes
        image_bytes = await file.read()

        # Reset file position for Cloudinary upload
        await file.seek(0)

        # Upload to Cloudinary for storage
        print("DEBUG: Starting Cloudinary upload...")
        cloud_result = cloudinary.uploader.upload(file.file, folder="skinai/scans")
        print("DEBUG: Upload successful!")
        image_url = cloud_result["secure_url"]

        # Run real ML inference
        print("DEBUG: Running ML prediction...")
        prediction = ml_predict(image_bytes)
        print(f"DEBUG: Prediction result: {prediction['disease']} ({prediction['confidence']:.2%})")

        # Save result to database
        scan = SkinScan(
            user_id=str(current_user.id),
            image_url=image_url,
            disease_detected=prediction["disease"],
            confidence_score=prediction["confidence"],
            severity_level=prediction["severity"],
            description=prediction["description"],
            recommendation=prediction["recommendation"],
            do_list=prediction.get("do_list", []),
            dont_list=prediction.get("dont_list", []),
        )
        await db.save(scan)

        # Return explicit dict (odmantic .dict() can have serialization quirks)
        return {
            "id": str(scan.id),
            "user_id": scan.user_id,
            "image_url": scan.image_url,
            "disease_detected": scan.disease_detected,
            "confidence_score": scan.confidence_score,
            "severity_level": scan.severity_level,
            "description": scan.description,
            "recommendation": scan.recommendation,
            "created_at": scan.created_at.isoformat() + "Z" if scan.created_at else None,
            "do_list": prediction.get("do_list", []),
            "dont_list": prediction.get("dont_list", [])
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_scan_history(
    current_user: User = Depends(get_current_user),
    db: AIOEngine = Depends(get_db)
):
    scans = await db.find(SkinScan, SkinScan.user_id == str(current_user.id), sort=SkinScan.created_at.desc())
    return [
        {
            "id": str(scan.id),
            "user_id": scan.user_id,
            "image_url": scan.image_url,
            "disease_detected": scan.disease_detected,
            "confidence_score": scan.confidence_score,
            "severity_level": scan.severity_level,
            "description": scan.description,
            "recommendation": scan.recommendation,
            "created_at": scan.created_at.isoformat() + "Z" if scan.created_at else None,
            "do_list": scan.do_list or [],
            "dont_list": scan.dont_list or []
        }
        for scan in scans
    ]
