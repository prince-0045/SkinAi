
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from odmantic import AIOEngine
from app.models.domain import SkinScan
from app.core.config import settings
import sys

# Windows asyncio policy fix
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import certifi

async def check_scans():
    print(f"Connecting to DB: {settings.MONGO_URL}")
    client = AsyncIOMotorClient(
        settings.MONGO_URL,
        tls=True,
        tlsAllowInvalidCertificates=True,
        tlsCAFile=certifi.where()
    )
    engine = AIOEngine(client=client, database="skincare_ai")
    
    # Fetch last 5 scans
    scans = await engine.find(SkinScan, sort=SkinScan.created_at.desc(), limit=5)
    
    print(f"\n--- Found {len(scans)} Recent Scans ---")
    if not scans:
        print("No scans found in database.")
    
    for scan in scans:
        print(f"ID: {scan.id}")
        print(f"Time: {scan.created_at}")
        print(f"URL: {scan.image_url}")
        print(f"Disease: {scan.disease_detected}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(check_scans())
