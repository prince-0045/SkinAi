
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import certifi

load_dotenv()
mongo_url = os.getenv("MONGO_URL")

print(f"Testing connection to: {mongo_url.split('@')[1] if '@' in mongo_url else 'encoded url'}")

try:
    print("Attempting to connect with certifi...")
    client = MongoClient(mongo_url, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("SUCCESS: Connected with certifi!")
except Exception as e:
    print(f"FAILED with certifi: {e}")

try:
    print("\nAttempting to connect with tlsAllowInvalidCertificates=True...")
    client = MongoClient(mongo_url, tlsAllowInvalidCertificates=True, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("SUCCESS: Connected with tlsAllowInvalidCertificates=True!")
except Exception as e:
    print(f"FAILED with tlsAllowInvalidCertificates=True: {e}")
