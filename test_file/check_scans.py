
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
    
    with open("scans_log.txt", "w") as f:
        f.write(f"\n--- Found {len(scans)} Recent Scans ---\n")
        if not scans:
            f.write("No scans found in database.\n")
        
        for scan in scans:
            f.write(f"ID: {scan.id}\n")
            f.write(f"User ID: {scan.user_id}\n")
            f.write(f"Time: {scan.created_at}\n")
            f.write(f"URL: {scan.image_url}\n")
            f.write(f"Disease: {scan.disease_detected}\n")
            f.write("-" * 30 + "\n")
    print("Log written to scans_log.txt")

if __name__ == "__main__":
    asyncio.run(check_scans())
