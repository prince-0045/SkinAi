import httpx # Use httpx for API Key access
from typing import List, Dict, Any
from app.core.config import settings

async def find_nearby_dermatologists(lat: float, lng: float, radius: int = 5000) -> List[Dict[str, Any]]:
    """
    Finds places near coordinates using Amazon Geo Places API (v2) with API Key.
    """
    api_key = settings.AWS_ACCESS_KEY_ID # User will paste API Key here
    region = settings.AWS_REGION
    
    if not api_key:
         return [
            {
                "name": "Skin Care Clinic (Mock)",
                "address": "123 AWS Blvd, Cloud City",
                "rating": None,
                "user_ratings_total": 0,
                "open_now": None,
                "place_id": "mock_aws_1",
                "geometry": {"location": {"lat": lat + 0.01, "lng": lng + 0.01}}
            }
        ]

    # Amazon Geo Places v2 Endpoint
    url = f"https://places.geo.{region}.amazonaws.com/v2/search-text"
    
    params = {
        "key": api_key 
    }
    
    # Body for POST request
    body = {
        "QueryText": "dermatologist",
        "BiasPosition": [lng, lat], # [lng, lat]
        "MaxResults": 5,
        "FilterCountries": ["IN"] # Optional: Filter by country (remove if global)
    }

    print(f"DEBUG: Searching AWS Place Index with Key={api_key[:5]}... Region={region}")

    try:
        async with httpx.AsyncClient() as client:
            print(f"DEBUG: sending request to {url}")
            response = await client.post(url, params=params, json=body)
            print(f"DEBUG: Response Status: {response.status_code}")
            
        if response.status_code != 200:
            print(f"AWS Geo Places API Error: {response.status_code} - {response.text}")
            return []

        data = response.json()
        results = data.get("ResultItems", []) # v2 returns ResultItems
        doctors = []
        
        for item in results:
            place = item # In v2, item IS the place details
            geometry = place.get("Position", [0, 0]) # [lng, lat]
            
            doctors.append({
                "name": place.get("Title", "Unknown Doctor"),
                "address": place.get("Address", {}).get("Label", ""),
                "rating": None, # Not available
                "user_ratings_total": 0,
                "open_now": None,
                "place_id": place.get("PlaceId", "unknown"),
                "geometry": {
                    "location": {
                        "lat": geometry[1], 
                        "lng": geometry[0]
                    }
                }
            })
            
        return doctors

    except Exception as e:
        print(f"AWS Location Service Error: {e}")
        return []
