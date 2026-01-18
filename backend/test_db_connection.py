
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
import sys

# Windows asyncio policy fix
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import certifi

async def test_connection():
    print(f"Testing connection to: {settings.MONGO_URL}")
    try:
        client = AsyncIOMotorClient(
            settings.MONGO_URL, 
            serverSelectionTimeoutMS=5000,
            tls=True,
            tlsAllowInvalidCertificates=True,
            tlsCAFile=certifi.where()
        )
        print("Client created. Ping server...")
        await client.server_info()
        print("SUCCESS: Connected to MongoDB!")
    except Exception as e:
        print(f"ERROR: Failed to connect to MongoDB.")
        print(f"Details: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_connection())
