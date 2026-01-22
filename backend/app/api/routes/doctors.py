from fastapi import APIRouter, Body, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from app.services.maps import find_nearby_dermatologists

router = APIRouter()

class Coordinates(BaseModel):
    latitude: float
    longitude: float

@router.post("/nearby")
async def get_nearby_doctors(coords: Coordinates):
    try:
        doctors = await find_nearby_dermatologists(coords.latitude, coords.longitude)
        return doctors
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
