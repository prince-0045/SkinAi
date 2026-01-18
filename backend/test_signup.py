import requests
import asyncio

URL = "http://localhost:8000/api/v1/auth/signup"

def test_signup():
    payload = {
        "name": "Test User",
        "email": "testuser_debug@example.com",
        "password": "password123"
    }
    
    print(f"Sending POST request to {URL}...")
    try:
        response = requests.post(URL, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 201:
            print("[PASS] Signup Successful!")
        elif response.status_code == 400 and "Email already registered" in response.text:
             print("[PASS] Signup Logic Reached (User already exists).")
        else:
            print("[FAIL] Signup Failed.")
            
    except Exception as e:
        print(f"[FAIL] Request Exception: {e}")

if __name__ == "__main__":
    test_signup()
