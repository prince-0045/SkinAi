from app.core.database import get_db
from app.models.user import User
import asyncio

async def check_users():
    db = await get_db()
    users = await db.find(User)
    print(f"Found {len(users)} users:")
    for u in users:
        print(f"\nUser: {u.email}")
        print(f"  Verified: {u.is_verified}")
        print(f"  Auth: {u.auth_provider}")
        print(f"  Password: {u.hashed_password}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(check_users())
