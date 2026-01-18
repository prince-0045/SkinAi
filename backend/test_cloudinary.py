
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import os

# Load env variables directly to avoid dependency on app.core.config for this test
load_dotenv(".env")

cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
api_key = os.getenv("CLOUDINARY_API_KEY")
api_secret = os.getenv("CLOUDINARY_API_SECRET")

print(f"DEBUG: Using Cloud Name: {cloud_name}")
print(f"DEBUG: API Key present: {bool(api_key)}")
print(f"DEBUG: API Secret present: {bool(api_secret)}")

cloudinary.config( 
  cloud_name = cloud_name, 
  api_key = api_key, 
  api_secret = api_secret 
)

try:
    print("Attempting to upload a test file...")
    # Create a dummy file
    with open("test_image.txt", "w") as f:
        f.write("test image content")
        
    result = cloudinary.uploader.upload("test_image.txt", resource_type="raw", folder="skinai_test")
    print("SUCCESS: Upload successful!")
    print(f"URL: {result.get('secure_url')}")
except Exception as e:
    print(f"ERROR: Upload failed.")
    print(f"Details: {e}")
finally:
    if os.path.exists("test_image.txt"):
        os.remove("test_image.txt")
