from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
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


# ── #4: Async Cloudinary upload (runs in background after prediction) ──
def _upload_to_cloudinary_sync(image_bytes: bytes, scan_id: str, db_engine_url: str):
    """Upload image to Cloudinary in background, update the DB record."""
    import io
    try:
        cloud_result = cloudinary.uploader.upload(
            io.BytesIO(image_bytes),
            folder="skinai/scans",
            resource_type="image"
        )
        image_url = cloud_result["secure_url"]
        # Update the DB record with the real URL (async via motor)
        import asyncio
        from motor.motor_asyncio import AsyncIOMotorClient
        from odmantic import AIOEngine
        from bson import ObjectId

        async def _update_url():
            from app.core.config import settings
            client = AsyncIOMotorClient(settings.MONGO_URL)
            engine = AIOEngine(client=client, database="test")
            collection = engine.get_collection(SkinScan)
            await collection.update_one(
                {"_id": ObjectId(scan_id)},
                {"$set": {"image_url": image_url}}
            )
            client.close()

        # Run the async update in a new event loop (we're in a background thread)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_update_url())
        loop.close()
    except Exception as e:
        print(f"[WARN] Background Cloudinary upload failed for scan {scan_id}: {e}")


@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AIOEngine = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        # Check daily upload limit
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            daily_scans_count = await db.count(
                SkinScan, 
                (SkinScan.user_id == str(current_user.id)) & (SkinScan.created_at >= today_start)
            )
        except Exception:
            daily_scans_count = 0
            
        if daily_scans_count >= 5:
            raise HTTPException(status_code=429, detail="Daily upload limit reached (5 scans max per day). Please try again tomorrow.")

        # Read the uploaded image bytes
        image_bytes = await file.read()

        # Run ML inference FIRST (return fast, upload later)
        prediction = ml_predict(image_bytes)

        # Save scan to DB immediately with placeholder image URL
        scan = SkinScan(
            user_id=str(current_user.id),
            image_url="pending",  # Will be updated by background task
            disease_detected=prediction["disease"],
            confidence_score=prediction["confidence"],
            severity_level=prediction["severity"],
            description=prediction["description"],
            recommendation=prediction["recommendation"],
            do_list=prediction.get("do_list", []),
            dont_list=prediction.get("dont_list", []),
        )
        await db.save(scan)

        # #4: Upload to Cloudinary in the background (doesn't block response)
        background_tasks.add_task(
            _upload_to_cloudinary_sync, image_bytes, str(scan.id), ""
        )

        # Return result immediately
        return {
            "id": str(scan.id),
            "user_id": scan.user_id,
            "image_url": scan.image_url,
            "disease_detected": scan.disease_detected,
            "category": prediction["category"],
            "confidence_score": scan.confidence_score,
            "severity_level": scan.severity_level,
            "is_unknown": prediction.get("is_unknown", False),
            "includes": prediction.get("includes", []),
            "description": scan.description,
            "recommendation": scan.recommendation,
            "created_at": scan.created_at.isoformat() + "Z" if scan.created_at else None,
            "do_list": prediction.get("do_list", []),
            "dont_list": prediction.get("dont_list", [])
        }
    except HTTPException:
        raise
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

@router.get("/limit")
async def get_upload_limit(
    current_user: User = Depends(get_current_user),
    db: AIOEngine = Depends(get_db)
):
    try:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        daily_scans_count = await db.count(
            SkinScan, 
            (SkinScan.user_id == str(current_user.id)) & (SkinScan.created_at >= today_start)
        )
    except Exception:
        daily_scans_count = 0
        
    remaining = max(0, 5 - daily_scans_count)
    return { "remaining": remaining, "limit": 5 }
