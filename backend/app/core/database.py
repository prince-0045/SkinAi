from motor.motor_asyncio import AsyncIOMotorClient
from odmantic import AIOEngine
from app.core.config import settings
import os

import certifi

class Database:
    client: AsyncIOMotorClient = None
    engine: AIOEngine = None

db = Database()

async def get_db():
    if db.engine is None:
        db.client = AsyncIOMotorClient(
            settings.MONGO_URL,
            tls=True,
            tlsCAFile=certifi.where()
        )
        db.engine = AIOEngine(client=db.client, database="skincare_ai")
    return db.engine
