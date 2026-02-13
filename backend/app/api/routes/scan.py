from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.models.user import User
from app.models.domain import SkinScan
from app.api.deps import get_current_user
from app.core.database import get_db
from odmantic import AIOEngine
import cloudinary.uploader
from app.core import cloudinary_config
from datetime import datetime
import tempfile
import numpy as np
import cv2

router = APIRouter()

# Disease class labels
DISEASE_LABELS = ["Eczema", "Psoriasis", "Melanoma", "Acne", "Benign Keratosis"]

# Lazy-loaded pipeline singleton
_pipeline = None

def _get_pipeline():
    """Lazy-load the hair removal pipeline."""
    global _pipeline
    if _pipeline is None:
        from app.ml.hair_removal_pipeline import HairRemovalPipeline
        _pipeline = HairRemovalPipeline()
    return _pipeline


def _predict_with_pipeline(image_bytes: bytes) -> tuple[str, float, str]:
    """
    Run the full hair-robust prediction pipeline.
    Falls back to classical-only mode if deep models aren't trained yet.
    """
    # Decode image
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")

    pipeline = _get_pipeline()
    result = pipeline.process(image)

    # Check if classifier model exists
    try:
        import torch
        from pathlib import Path
        from app.ml.models.robust_classifier import RobustClassifier
        from app.ml.pipeline_config import PipelineConfig
        from app.ml.models.channel_composer import compose_to_tensor

        cfg = PipelineConfig()
        ckpt = Path(cfg.training.checkpoint_dir) / "best_classifier.pth"
        if ckpt.exists():
            model = RobustClassifier(cfg.classifier)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model.eval()
            tensor = compose_to_tensor(
                image, result["hair_mask"], result["directional"], result["frequency"]
            )
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(cfg.resolution.classifier_size, cfg.resolution.classifier_size)
            )
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
            idx = probs.argmax().item()
            confidence = probs[idx].item()
            disease = DISEASE_LABELS[idx] if idx < len(DISEASE_LABELS) else f"Class_{idx}"
            severity = "Severe" if confidence > 0.85 else "Moderate" if confidence > 0.6 else "Mild"
            return disease, confidence, severity
    except Exception:
        pass

    # Fallback: hair mask coverage heuristic
    hair_pct = result["hair_mask"].sum() / (result["hair_mask"].size * 255) * 100
    return "Analysis Pending (model not trained)", 0.0, f"Hair coverage: {hair_pct:.1f}%"


@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AIOEngine = Depends(get_db)
):
    try:
        # Save file to Cloudinary
        contents = await file.read()
        await file.seek(0)
        result = cloudinary.uploader.upload(file.file, folder="skinai/scans")
        image_url = result["secure_url"]

        # Run ML pipeline
        disease, confidence, severity = _predict_with_pipeline(contents)

        scan = SkinScan(
            user_id=str(current_user.id),
            image_url=image_url,
            disease_detected=disease,
            confidence_score=confidence,
            severity_level=severity,
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
