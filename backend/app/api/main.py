from fastapi import APIRouter
from app.api.routes import auth, users, scan, doctors

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Auth"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(scan.router, prefix="/scan", tags=["Scan"])
api_router.include_router(doctors.router, prefix="/doctors", tags=["Doctors"])


