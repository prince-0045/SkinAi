# API REFERENCE

Complete reference of every API endpoint in the SkinAI backend. Organized by feature/route.

---

## Base URL

```
Local Development: http://localhost:8000/api/v1
Production: https://dermaura.tech/api/v1
```

---

## Authentication Endpoints

### 1. User Signup

**Endpoint:** `POST /auth/signup`

**Description:** Create a new user account with email and password. Returns JWT access token immediately (direct login).

**Request:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "MyPassword123"
}
```

**Query/Path Parameters:** None

**Headers Required:** `Content-Type: application/json`

**Response (201 Created):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "name": "John Doe",
    "email": "john@example.com",
    "created_at": "2025-03-31T12:00:00"
  }
}
```

**Error Responses:**
- `400 Bad Request` — Email already registered
- `400 Bad Request` — Password too short / missing uppercase / missing number
- `500 Internal Server Error` — Password hashing failed

**Side Effects:** Creates User record in MongoDB, may send welcome email

**Handler Function:** `backend/app/api/routes/auth.py` → `signup()`

---

### 2. User Login

**Endpoint:** `POST /auth/login`

**Description:** Authenticate user with email/password. Implements brute force protection (5 attempts = 5 min lockout).

**Request:**
```json
{
  "email": "john@example.com",
  "password": "MyPassword123"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "name": "John Doe",
    "email": "john@example.com",
    "created_at": "2025-03-31T12:00:00"
  }
}
```

**Error Responses:**
- `401 Unauthorized` — Invalid credentials (email not found or wrong password)
- `401 Unauthorized` — Please use Google Sign-In (for Google-auth users)
- `429 Too Many Requests` — Too many login attempts. Please try again in 5 minutes.

**Side Effects:** Updates user.last_login timestamp, records failed attempts in memory

**Handler Function:** `backend/app/api/routes/auth.py` → `login()`

---

### 3. Google OAuth Login

**Endpoint:** `POST /auth/google`

**Description:** Authenticate user via Google OAuth token. Validates token with Google API, creates/finds user.

**Request:**
```json
{
  "token": "ya29.a0AfH6SMBx..."
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "name": "John Doe",
    "email": "john@example.com",
    "created_at": "2025-03-31T12:00:00"
  }
}
```

**Error Responses:**
- `401 Unauthorized` — Invalid Google Access Token
- `400 Bad Request` — Google Account email not found

**Side Effects:** Creates new User if doesn't exist, updates auth_provider to "google", issues JWT

**Handler Function:** `backend/app/api/routes/auth.py` → `google_login()`

---

### 4. Test Email Endpoint (Diagnostic)

**Endpoint:** `GET /auth/test-email?email=john@example.com`

**Description:** Diagnostic endpoint to test email delivery. Sends test OTP and returns detailed error info.

**Query Parameters:**
- `email` (string, required) — Email to send test to

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Test email sent to john@example.com"
}
```

**Error Response:**
```json
{
  "status": "error",
  "error_type": "SMTPException",
  "error_detail": "421 mta7.iad.mail.yahoo.com...",
  "traceback": "Traceback (most recent call last)..."
}
```

**Side Effects:** Sends test email via Resend API, does NOT save to database

**Handler Function:** `backend/app/api/routes/auth.py` → `test_email_diag()`

---

## User Management Endpoints

### 5. Get Current User

**Endpoint:** `GET /users/me`

**Description:** Retrieve current authenticated user's profile information.

**Authentication:** Required (Bearer JWT in Authorization header)

**Request:** No body

**Response (200 OK):**
```json
{
  "id": "507f1f77bcf86cd799439011",
  "name": "John Doe",
  "email": "john@example.com",
  "hashed_password": "$argon2id...",
  "auth_provider": "email",
  "is_verified": true,
  "created_at": "2025-03-31T12:00:00",
  "last_login": "2025-03-31T15:30:00"
}
```

**Error Responses:**
- `401 Unauthorized` — Invalid or missing JWT token
- `401 Unauthorized` — Email not verified (commented out)

**Side Effects:** None (read-only)

**Handler Function:** `backend/app/api/routes/users.py` → `read_users_me()`

---

### 6. Change Password

**Endpoint:** `POST /users/change-password`

**Description:** Change user's password. Validates current password and enforces password policy.

**Authentication:** Required

**Request:**
```json
{
  "current_password": "OldPassword123",
  "new_password": "NewPassword456"
}
```

**Response (200 OK):**
```json
{
  "message": "Password updated successfully"
}
```

**Error Responses:**
- `400 Bad Request` — Incorrect current password
- `400 Bad Request` — New password must be at least 8 characters
- `400 Bad Request` — New password must contain at least one uppercase letter
- `400 Bad Request` — New password must contain at least one number
- `401 Unauthorized` — Invalid JWT token

**Side Effects:** Updates user.hashed_password in MongoDB using Argon2

**Handler Function:** `backend/app/api/routes/users.py` → `change_password()`

---

### 7. Delete Account

**Endpoint:** `DELETE /users/account`

**Description:** Permanently delete user account and ALL associated scan data. This is irreversible.

**Authentication:** Required

**Request:** No body

**Response (200 OK):**
```json
{
  "message": "Account and all associated data deleted successfully"
}
```

**Error Responses:**
- `401 Unauthorized` — Invalid JWT token
- `500 Internal Server Error` — Failed to delete account. Please try again.

**Side Effects:** 
- Deletes all SkinScan records for user from MongoDB
- Deletes User record from MongoDB
- Logs deletion

**Handler Function:** `backend/app/api/routes/users.py` → `delete_account()`

---

### 8. User Pulse (Heartbeat)

**Endpoint:** `POST /users/pulse`

**Description:** Update user's active session timestamp. Called every 60 seconds to track real-time online users.

**Authentication:** Required

**Request:** No body

**Response (200 OK):**
```json
{
  "status": "ok"
}
```

**Error Responses:**
- `401 Unauthorized` — Invalid JWT token

**Side Effects:** 
- Creates or updates ActiveSession record in MongoDB
- Sets last_seen_at to current UTC time
- Used by admin dashboard to track live users

**Handler Function:** `backend/app/api/routes/users.py` → `user_pulse()`

---

## Skin Scan / ML Endpoints

### 9. Predict Skin Disease

**Endpoint:** `POST /scan/predict`

**Description:** Upload skin image and get AI-powered disease prediction. Main feature endpoint.

**Authentication:** Optional (commented out)

**Request:** Multipart form-data

**Request Body:**
```
POST /scan/predict HTTP/1.1
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="file"; filename="skin_photo.jpg"
Content-Type: image/jpeg

[binary image data]
------WebKitFormBoundary--
```

**Headers:**
- `Authorization: Bearer <JWT>` (optional, currently unused)

**Response (200 OK):**
```json
{
  "id": "507f1f77bcf86cd799439012",
  "user_id": "507f1f77bcf86cd799439011",
  "image_url": "pending",
  "disease_detected": "Acne",
  "category": "Acne",
  "confidence_score": 0.92,
  "severity_level": "Mild",
  "is_unknown": false,
  "includes": [],
  "description": "A common skin condition where hair follicles become clogged with oil and dead skin cells...",
  "recommendation": "Use a gentle cleanser and consider over-the-counter treatments...",
  "created_at": "2025-03-31T12:00:00Z",
  "do_list": [
    "Wash affected areas gently twice daily",
    "Use non-comedogenic skincare products",
    "Apply prescribed topical treatments consistently"
  ],
  "dont_list": [
    "Do not pop, squeeze, or pick at pimples",
    "Avoid harsh scrubbing or over-washing",
    "Do not use oily or greasy cosmetics"
  ]
}
```

**Error Responses:**
- `400 Bad Request` — Invalid file type (must be JPEG, PNG, or WebP)
- `400 Bad Request` — File too large (max 10MB)
- `400 Bad Request` — Invalid image file
- `429 Too Many Requests` — Daily upload limit reached (5 scans max per day)
- `500 Internal Server Error` — Analysis failed

**File Constraints:**
- **Allowed types:** `image/jpeg`, `image/png`, `image/webp`
- **Max size:** 10MB
- **Min size:** No minimum
- **Image validation:** PIL.Image.open().verify() must succeed

**Side Effects:**
- Runs expensive ML inference (MobileNetV2 model)
- Compresses image client-side before sending
- Asynchronously uploads to Cloudinary CDN in background (updates DB later)
- Currently disabled: Saves to MongoDB (needs re-enabling)

**ML Model Details:**
- Model: MobileNetV2 fine-tuned for skin disease
- Input: 224x224 pixels RGB
- Output: 13 disease classes
- Confidence threshold: 35% (returns "Unknown" if below)

**Handler Function:** `backend/app/api/routes/scan.py` → `predict_disease()`

---

### 10. Get Scan History

**Endpoint:** `GET /scan/history`

**Description:** Retrieve all of user's past skin scans, sorted by most recent first.

**Authentication:** Required

**Query Parameters:** None

**Response (200 OK):**
```json
[
  {
    "id": "507f1f77bcf86cd799439012",
    "user_id": "507f1f77bcf86cd799439011",
    "image_url": "https://res.cloudinary.com/...",
    "disease_detected": "Acne",
    "confidence_score": 0.92,
    "severity_level": "Mild",
    "description": "...",
    "recommendation": "...",
    "created_at": "2025-03-31T15:30:00Z",
    "do_list": [...],
    "dont_list": [...]
  },
  {
    "id": "507f1f77bcf86cd799439013",
    "user_id": "507f1f77bcf86cd799439011",
    "image_url": "https://res.cloudinary.com/...",
    "disease_detected": "Eczema",
    "confidence_score": 0.87,
    "severity_level": "Moderate",
    "description": "...",
    "recommendation": "...",
    "created_at": "2025-03-31T10:00:00Z",
    "do_list": [...],
    "dont_list": [...]
  }
]
```

**Error Responses:**
- `401 Unauthorized` — Invalid JWT token

**Side Effects:** None (read-only query)

**Handler Function:** `backend/app/api/routes/scan.py` → `get_scan_history()`

---

### 11. Get Upload Limit

**Endpoint:** `GET /scan/limit`

**Description:** Get remaining daily upload limit for current user. Limit is 5 scans per day, resets at midnight UTC.

**Authentication:** Required

**Request:** No body

**Response (200 OK):**
```json
{
  "remaining": 3,
  "limit": 5
}
```

**Error Responses:**
- `401 Unauthorized` — Invalid JWT token

**Side Effects:** Counts user's scans from today in MongoDB to calculate remaining

**Handler Function:** `backend/app/api/routes/scan.py` → `get_upload_limit()`

---

## Doctor Finding Endpoints

### 12. Find Nearby Doctors

**Endpoint:** `POST /doctors/nearby`

**Description:** Find dermatologists near user's GPS coordinates using AWS Location Services.

**Authentication:** Not required

**Request:**
```json
{
  "latitude": 40.7128,
  "longitude": -74.0060
}
```

**Response (200 OK):**
```json
[
  {
    "name": "Dr. Sarah Johnson Dermatology",
    "address": "123 Medical Center Dr, New York, NY 10001",
    "rating": 4.8,
    "user_ratings_total": 256,
    "open_now": true,
    "place_id": "ChIJN1blbx",
    "geometry": {
      "location": {
        "lat": 40.7138,
        "lng": -74.0055
      }
    }
  },
  {
    "name": "Manhattan Dermatology Clinic",
    "address": "456 Park Ave, New York, NY 10022",
    "rating": 4.5,
    "user_ratings_total": 189,
    "open_now": true,
    "place_id": "ChIJN1blbx",
    "geometry": {
      "location": {
        "lat": 40.7630,
        "lng": -73.9776
      }
    }
  }
]
```

**Mock Response (if API key not set):**
```json
[
  {
    "name": "Skin Care Clinic (Mock)",
    "address": "123 AWS Blvd, Cloud City",
    "rating": null,
    "user_ratings_total": 0,
    "open_now": null,
    "place_id": "mock_aws_1",
    "geometry": {
      "location": {
        "lat": 40.7238,
        "lng": -73.9960
      }
    }
  }
]
```

**Error Responses:**
- `500 Internal Server Error` — AWS Location API error

**Search Parameters:**
- **Search query:** "dermatologist"
- **Radius:** 5000 meters (configurable)
- **Max results:** 5 doctors
- **Filter:** India (optional, can be removed)

**Side Effects:** Calls AWS Location Services API (external service)

**Handler Function:** `backend/app/api/routes/doctors.py` → `get_nearby_doctors()`

---

## Admin Endpoints

### 13. Get Admin Statistics

**Endpoint:** `GET /admin/stats`

**Description:** Get system-wide statistics. Admin-only endpoint protected by password header.

**Authentication:** Required header `x-admin-password`

**Headers Required:**
```
x-admin-password: <admin_password_from_env>
```

**Request:** No body

**Response (200 OK):**
```json
{
  "total_users": 1250,
  "active_users_24h": 345,
  "live_users": 28,
  "total_scans": 4567,
  "recent_users": [
    {
      "name": "John Doe",
      "email": "john@example.com",
      "created_at": "2025-03-31T15:30:00"
    }
  ],
  "recent_scans": [
    {
      "disease": "Acne",
      "confidence": 0.92,
      "created_at": "2025-03-31T15:45:00"
    }
  ]
}
```

**Error Responses:**
- `401 Unauthorized` — Invalid or missing admin password

**Statistics Definitions:**
- **total_users:** All users ever registered
- **active_users_24h:** Users who logged in within last 24 hours
- **live_users:** Users who sent heartbeat within last 2 minutes
- **total_scans:** All skin scans ever submitted
- **recent_users:** Last 10 registered users
- **recent_scans:** Last 10 scans submitted

**Side Effects:** Multiple read-only database queries

**Security Note:** [SECURITY ISSUE] Uses hardcoded password from environment, no proper OAuth

**Handler Function:** `backend/app/api/routes/admin.py` → `get_admin_stats()`

---

## Health Check Endpoints

### 14. Health Check Endpoint

**Endpoint:** `GET /health`

**Description:** Health check for load balancers and monitoring systems. Returns status of database and ML model.

**Authentication:** Not required

**Request:** No body

**Response (200 OK):**
```json
{
  "status": "healthy",
  "database": "connected",
  "ml_model": "loaded",
  "timestamp": "2025-03-31T15:45:00Z"
}
```

**Database Status Values:**
- `connected` — MongoDB connection working
- `not_initialized` — Database not yet initialized
- `disconnected` — MongoDB connection failed

**ML Model Status Values:**
- `loaded` — Model cached in memory
- `not_loaded` — Model not yet loaded (will lazy-load on first prediction)

**Side Effects:** Pings MongoDB, checks model cache (no data modifications)

**Handler Function:** `backend/app/main.py` → `health_check()`

---

### 15. Root Endpoint

**Endpoint:** `GET /` (no prefix)

**Description:** Simple root endpoint, returns welcome message.

**Response (200 OK):**
```json
{
  "message": "Welcome to SkinCare AI API"
}
```

**Handler Function:** `backend/app/main.py` → `root()`

---

## Rate Limiting

**Global Rate Limit:** 1000 requests per minute per IP address

**Rate Limit Headers:**
- `X-RateLimit-Limit: 1000`
- `X-RateLimit-Remaining: 999`
- `X-RateLimit-Reset: 1609459200`

**Rate Limit Exceeded Response (429):**
```json
{
  "detail": "Too many requests. Please slow down and try again shortly."
}
```

---

## Error Response Format

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common Status Codes:**
- `200 OK` — Success
- `201 Created` — Resource created
- `400 Bad Request` — Invalid input/validation error
- `401 Unauthorized` — Authentication required or failed
- `429 Too Many Requests` — Rate limit exceeded
- `500 Internal Server Error` — Unexpected server error

---

## Authentication

### Bearer Token Usage

All authenticated endpoints require the JWT token in the Authorization header:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Token Expiry

- **Expiry:** 24 hours from creation
- **Refresh:** Obtain new token by login again

### Token Validation

- **Algorithm:** HS256
- **Secret:** `settings.SECRET_KEY` from environment
- **Claims:** `{sub: email, exp: timestamp, iat: timestamp}`

---

## CORS Configuration

**Allowed Origins:**
- `http://localhost:5173` (dev frontend)
- `http://localhost:3000` (alt dev)
- `https://dermaura.tech`
- `https://www.dermaura.tech`
- Any subdomain: `https://*.dermaura.tech` (regex)

**Allowed Methods:** GET, POST, PUT, DELETE, OPTIONS

**Allowed Headers:** All (*)

**Allowed Credentials:** Yes (cookies supported)

---

## Summary Table

| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| POST | /auth/signup | ✗ | Register new user |
| POST | /auth/login | ✗ | Login with credentials |
| POST | /auth/google | ✗ | Login with Google |
| GET | /auth/test-email | ✗ | Test email delivery |
| GET | /users/me | ✓ | Get current user |
| POST | /users/change-password | ✓ | Change password |
| DELETE | /users/account | ✓ | Delete account |
| POST | /users/pulse | ✓ | Send heartbeat |
| POST | /scan/predict | ~ | Scan image (optional) |
| GET | /scan/history | ✓ | Get scan history |
| GET | /scan/limit | ✓ | Get daily limit |
| POST | /doctors/nearby | ✗ | Find doctors |
| GET | /admin/stats | ~password | Admin statistics |
| GET | /health | ✗ | Health check |
| GET | / | ✗ | Root message |

Legend: ✓ = Required, ✗ = Not required, ~ = Special auth

