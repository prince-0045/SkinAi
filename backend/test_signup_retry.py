import requests
import asyncio

URL_SIGNUP = "http://localhost:8000/api/v1/auth/signup"

def test_signup_retry():
    # Use the specific email user is struggling with
    payload = {
        "name": "Shrey Test",
        "email": "shreypatel0605@gmail.com",
        "password": "newpassword123"
    }
    
    print(f"Attempting Signup for {payload['email']} (should work even if exists & unverified)...")
    try:
        response = requests.post(URL_SIGNUP, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 201:
            print("[PASS] Signup Successful (Old unverified user deleted/overwritten)!")
        else:
            print("[FAIL] Signup Failed.")
            
    except Exception as e:
        print(f"[FAIL] Request Exception: {e}")

if __name__ == "__main__":
    test_signup_retry()
