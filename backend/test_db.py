import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import certifi
import os

# Updated URL without the invalid cert flag
MONGO_URL = "mongodb+srv://pilu_skinai:pilu%402001@cluster0.wmgxzy3.mongodb.net/?appName=Cluster0"

async def test_connection():
    print(f"Testing connection to: {MONGO_URL.split('@')[1]}")
    try:
        # Replicating the fixed logic
        client = AsyncIOMotorClient(
            MONGO_URL,
            tls=True,
            # tlsAllowInvalidCertificates=True,  <-- REMOVED
            tlsCAFile=certifi.where()
        )
        print("Client created.")
        
        print("Pinging server...")
        await client.admin.command('ping')
        print("Ping successful!")
        
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
