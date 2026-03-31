import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os

# Using the URL with the bypass flag
MONGO_URL = "mongodb+srv://pilu_skinai:pilu%402001@cluster0.wmgxzy3.mongodb.net/?appName=Cluster0&tlsAllowInvalidCertificates=true"

async def test_connection():
    print(f"Testing INSECURE connection")
    try:
        # Pymongo 4.x / Motor 3.x
        # We try to force insecure mode without providing a CA file
        client = AsyncIOMotorClient(
            MONGO_URL,
            tls=True,
            tlsAllowInvalidCertificates=True
            # NO tlsCAFile
        )
        print("Client created.")
        
        print("Pinging server...")
        await client.admin.command('ping')
        print("Ping successful!")
        
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
