import asyncio
import sys
from app.core.database import get_db
from app.models.domain import SkinScan
from odmantic import AIOEngine
from bson import ObjectId

async def main():
    db: AIOEngine = await get_db()
    print("DB connection successful.")
    # just find any user's scans
    try:
        scans = await db.find(SkinScan, sort=SkinScan.created_at.desc())
        print(f"Found {len(scans)} scans")
        if not scans:
            print("No scans found")
            return
            
        print("First scan:", scans[0])
        # Try to map them like the endpoint does
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
                "created_at": scan.created_at.isoformat() if scan.created_at else None,
                "do_list": getattr(scan, 'do_list', []),
                "dont_list": getattr(scan, 'dont_list', [])
            }
            for scan in scans
        ]
        print("Mapping successful!", result[0])
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
