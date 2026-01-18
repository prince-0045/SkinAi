import asyncio
import os
import sys
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load env
load_dotenv()

async def check():
    print("--- Diagnostic Check ---")
    
    # 1. Check Requests
    try:
        import requests
        print("[OK] 'requests' library is installed.")
    except ImportError:
        print("[FAIL] 'requests' library is MISSING.")
        return

    # 2. Check MongoDB
    mongo_url = os.getenv("MONGO_URL")
    print(f"Checking MongoDB Connection to: {mongo_url}")
    
    try:
        client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=5000)
        # Force a command to check connection
        await client.admin.command('ping')
        print("[OK] MongoDB Connection Successful!")
    except Exception as e:
        print(f"[FAIL] MongoDB Connection Failed: {e}")
        print("Possible causes: IP not whitelisted in Atlas, wrong password, or firewall.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(check())
