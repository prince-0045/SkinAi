from fastapi import APIRouter, Depends, Body, HTTPException, status
from app.models.user import User
from app.models.domain import SkinScan
from app.api.deps import get_current_user
from app.core.auth import verify_password, get_password_hash
from app.core.database import get_db
from app.core.constants import PASSWORD_MIN_LENGTH
from odmantic import AIOEngine
import re
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def validate_password(password: str):
    """Enforce password strength requirements."""
    if len(password) < PASSWORD_MIN_LENGTH:
        raise HTTPException(400, f"Password must be at least {PASSWORD_MIN_LENGTH} characters")
    if not re.search(r'[A-Z]', password):
        raise HTTPException(400, "Password must contain at least one uppercase letter")
    if not re.search(r'[0-9]', password):
        raise HTTPException(400, "Password must contain at least one number")


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.post("/change-password")
async def change_password(
    current_password: str = Body(...),
    new_password: str = Body(...),
    current_user: User = Depends(get_current_user),
    db: AIOEngine = Depends(get_db)
):
    if not verify_password(current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )

    validate_password(new_password)

    current_user.hashed_password = get_password_hash(new_password)
    await db.save(current_user)

    return {"message": "Password updated successfully"}


@router.delete("/account")
async def delete_account(
    current_user: User = Depends(get_current_user),
    db: AIOEngine = Depends(get_db)
):
    """Delete user account and all associated scan data."""
    try:
        # Delete all user's scans
        scans = await db.find(SkinScan, SkinScan.user_id == str(current_user.id))
        for scan in scans:
            await db.delete(scan)

        # Delete user
        await db.delete(current_user)

        logger.info(f"Account deleted: {current_user.email}")
        return {"message": "Account and all associated data deleted successfully"}
    except Exception as e:
        logger.error(f"Account deletion failed for {current_user.email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete account. Please try again.")
