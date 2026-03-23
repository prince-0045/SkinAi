from motor.motor_asyncio import AsyncIOMotorClient
from odmantic import AIOEngine
from app.core.config import settings
import os

import certifi

class Database:
    client: AsyncIOMotorClient = None
    engine: AIOEngine = None

    def setup_db(self, db_url: str):
        """
        Initialize the database connection pool.
        Configured with connection pooling specifically for handling parallel
        machine learning inference logic and user background tasks.
        """
        if "mongodb+srv://" in db_url:
            # Atlas connection
            self.client = AsyncIOMotorClient(
                db_url,
                maxPoolSize=50,
                minPoolSize=10,
                serverSelectionTimeoutMS=5000,
                tls=True,
                tlsCAFile=certifi.where()
            )
        else:
            # Local connection
            self.client = AsyncIOMotorClient(
                db_url,
                maxPoolSize=50,
                minPoolSize=10
            )

db = Database()

async def get_db():
    if db.engine is None:
        db.setup_db(settings.MONGO_URL)
        db.engine = AIOEngine(client=db.client, database="skincare_ai")
    return db.engine
