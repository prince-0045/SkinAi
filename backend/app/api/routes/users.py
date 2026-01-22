from fastapi import APIRouter, Depends, Body, HTTPException, status
from app.models.user import User
from app.api.deps import get_current_user
from app.core.auth import verify_password, get_password_hash
from app.core.database import get_db
from odmantic import AIOEngine

router = APIRouter()

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
    
    current_user.hashed_password = get_password_hash(new_password)
    await db.save(current_user)
    
    return {"message": "Password updated successfully"}
