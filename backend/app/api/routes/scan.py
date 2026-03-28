from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from app.models.user import User
from app.models.domain import SkinScan
from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.constants import DAILY_SCAN_LIMIT, MAX_UPLOAD_SIZE_BYTES, ALLOWED_IMAGE_TYPES
from odmantic import AIOEngine
import cloudinary.uploader
from app.core import cloudinary_config
from app.services.ml_model import predict as ml_predict
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Background Cloudinary upload ──
def _upload_to_cloudinary_sync(image_bytes: bytes, scan_id: str):
    """Upload image to Cloudinary in background, update the DB record."""
    import io
    try:
        cloud_result = cloudinary.uploader.upload(
            io.BytesIO(image_bytes),
            folder="skinai/scans",
            resource_type="image"
        )
        image_url = cloud_result["secure_url"]

        # Update DB with real URL
        import asyncio as _asyncio
        from motor.motor_asyncio import AsyncIOMotorClient
        from odmantic import AIOEngine as _AIOEngine
        from bson import ObjectId

        async def _update_url():
            from app.core.config import settings
            client = AsyncIOMotorClient(settings.MONGO_URL)
            engine = _AIOEngine(client=client, database="skincare_ai")
            collection = engine.get_collection(SkinScan)
            await collection.update_one(
                {"_id": ObjectId(scan_id)},
                {"$set": {"image_url": image_url}}
            )
            client.close()

        loop = _asyncio.new_event_loop()
        loop.run_until_complete(_update_url())
        loop.close()
    except Exception as e:
        logger.warning(f"Background Cloudinary upload failed for scan {scan_id}: {e}")


@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AIOEngine = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        # ── File validation ──
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Please upload a JPEG, PNG, or WebP image."
            )

        image_bytes = await file.read()

        if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
            size_mb = len(image_bytes) / (1024 * 1024)
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({size_mb:.1f}MB). Maximum allowed is {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB."
            )

        # Verify it's a valid image
        try:
            from PIL import Image
            from io import BytesIO
            Image.open(BytesIO(image_bytes)).verify()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid photo.")

        # ── Check daily upload limit ──
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            daily_scans_count = await db.count(
                SkinScan,
                (SkinScan.user_id == str(current_user.id)) & (SkinScan.created_at >= today_start)
            )
        except Exception:
            daily_scans_count = 0

        if daily_scans_count >= DAILY_SCAN_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"Daily upload limit reached ({DAILY_SCAN_LIMIT} scans max per day). Please try again tomorrow."
            )

        # ── Run ML inference in thread pool (non-blocking) ──
        prediction = await asyncio.to_thread(ml_predict, image_bytes)

        # ── Save to DB with placeholder URL ──
        scan = SkinScan(
            user_id=str(current_user.id),
            image_url="pending",
            disease_detected=prediction["disease"],
            confidence_score=prediction["confidence"],
            severity_level=prediction["severity"],
            description=prediction["description"],
            recommendation=prediction["recommendation"],
            do_list=prediction.get("do_list", []),
            dont_list=prediction.get("dont_list", []),
        )
        await db.save(scan)

        # ── Upload to Cloudinary in background ──
        background_tasks.add_task(_upload_to_cloudinary_sync, image_bytes, str(scan.id))

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
    result = [
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
    return result

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

    remaining = max(0, DAILY_SCAN_LIMIT - daily_scans_count)
    return {"remaining": remaining, "limit": DAILY_SCAN_LIMIT}
