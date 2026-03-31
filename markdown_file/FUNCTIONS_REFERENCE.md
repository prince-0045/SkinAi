# FUNCTIONS REFERENCE

Complete reference of every function in the SkinAI codebase organized by file.

---

## Backend Functions

### `backend/app/main.py` - FastAPI Application Setup

---

**`rate_limit_handler()`** — `backend/app/main.py`

**What it does:** Handles rate limiting errors when users exceed request limits (1000 per minute). Returns a 429 HTTP error.

**Parameters:** `request` (Request), `exc` (RateLimitExceeded)

**Returns:** JSONResponse with status 429 and message "Too many requests"

**Side effects:** HTTP response returned to client, no database changes

**Used by:** SlowAPI middleware (automatic)

---

**`http_exception_handler()`** — `backend/app/main.py`

**What it does:** Catches HTTP exceptions and ensures CORS headers are included in error responses, fixing a common issue where errors were missing CORS headers.

**Parameters:** `request` (Request), `exc` (HTTPException)

**Returns:** JSONResponse with error details and CORS headers

**Side effects:** Sets CORS response headers

**Used by:** FastAPI exception handling (automatic)

---

**`log_response_time()`** — `backend/app/main.py`

**What it does:** Middleware that logs the time taken for each HTTP request and adds timing information to response headers.

**Parameters:** `request`, `call_next` (middleware parameter)

**Returns:** Response with added X-Response-Time header

**Side effects:** Logs to console/logger, adds response time header

**Used by:** FastAPI middleware (automatic)

---

**`startup_event()`** — `backend/app/main.py`

**What it does:** Runs on application startup. Pre-loads the ML model from disk into memory to avoid slow first request.

**Parameters:** None

**Returns:** None (async)

**Side effects:** Loads ML model into memory, logs status

**Used by:** FastAPI lifecycle events (automatic on app start)

---

**`shutdown_event()`** — `backend/app/main.py`

**What it does:** Runs on application shutdown. Closes MongoDB connection gracefully.

**Parameters:** None

**Returns:** None (async)

**Side effects:** Closes MongoDB client connection

**Used by:** FastAPI lifecycle events (automatic on app stop)

---

**`health_check()`** — `backend/app/main.py`

**What it does:** Health check endpoint for load balancers and monitoring. Returns status of database and ML model.

**Parameters:** None

**Returns:** 
```json
{
  "status": "healthy",
  "database": "connected|not_initialized|disconnected",
  "ml_model": "loaded|not_loaded",
  "timestamp": "2025-03-31T12:00:00Z"
}
```

**Side effects:** Pings MongoDB, checks model cache

**Used by:** Load balancers, container orchestration (Kubernetes)

---

### `backend/app/core/auth.py` - Authentication Utilities

---

**`verify_password()`** — `backend/app/core/auth.py`

**What it does:** Compares a plain text password against a stored hash. Supports both Argon2 and Bcrypt hashes with fallback for legacy plain text passwords.

**Parameters:** `plain_password` (str), `hashed_password` (str)

**Returns:** Boolean (True if password matches, False otherwise)

**Side effects:** None

**Used by:** `login()` in routes/auth.py, `change_password()` in routes/users.py

---

**`get_password_hash()`** — `backend/app/core/auth.py`

**What it does:** Hashes a plain text password using Argon2 algorithm. Creates a one-way hash that cannot be reversed.

**Parameters:** `password` (str)

**Returns:** String (hashed password with salt)

**Side effects:** None (pure function)

**Used by:** `signup()` in routes/auth.py, `change_password()` in routes/users.py

---

**`create_access_token()`** — `backend/app/core/auth.py`

**What it does:** Creates a JWT access token signed with the secret key. Token expires in 24 hours by default.

**Parameters:** 
- `data` (dict) — Data to encode in token (e.g., {"sub": "user@email.com"})
- `expires_delta` (timedelta, optional) — Custom expiry time

**Returns:** String (encoded JWT token)

**Side effects:** None (pure function)

**Used by:** `signup()`, `login()`, `google_login()` in routes/auth.py, `googleLogin()` in frontend

---

### `backend/app/core/database.py` - Database Connection

---

**`Database.setup_db()`** — `backend/app/core/database.py`

**What it does:** Initializes MongoDB connection with connection pooling. Handles both cloud (Atlas) and local MongoDB.

**Parameters:** `db_url` (str) — MongoDB connection string

**Returns:** None (sets `self.client` internally)

**Side effects:** Creates async MongoDB client, establishes connection pool

**Used by:** `get_db()` dependency injection

---

**`get_db()`** — `backend/app/core/database.py`

**What it does:** Dependency injection function that returns the ODMantic engine for database operations. Creates engine once and reuses it.

**Parameters:** None (FastAPI dependency)

**Returns:** AIOEngine instance (async ODM engine)

**Side effects:** Calls `setup_db()` on first call, caches result

**Used by:** Every route that needs database access (via `Depends(get_db)`)

---

### `backend/app/api/deps.py` - Dependency Injection

---

**`get_current_user()`** — `backend/app/api/deps.py`

**What it does:** Extracts JWT token from Authorization header, validates it, and returns the current authenticated user. Core dependency for protecting routes.

**Parameters:** 
- `token` (str) — Bearer token from Authorization header (dependency)
- `db` (AIOEngine) — Database engine (dependency)

**Returns:** User object (from MongoDB)

**Raises:** HTTPException 401 if token is invalid or expired

**Side effects:** Queries MongoDB for user by email

**Used by:** `read_users_me()`, `change_password()`, `delete_account()`, `predict_disease()`, `get_scan_history()`, etc.

---

### `backend/app/api/routes/auth.py` - Authentication Routes

---

**`generate_otp()`** — `backend/app/api/routes/auth.py`

**What it does:** Generates a random 6-digit OTP (One-Time Password) for email verification.

**Parameters:** None

**Returns:** String of 6 digits (e.g., "123456")

**Side effects:** None (uses random module)

**Used by:** OTP endpoints (if implemented)

---

**`validate_password()`** — `backend/app/api/routes/auth.py`

**What it does:** Validates that password meets security requirements: 8+ chars, at least 1 uppercase letter, at least 1 number.

**Parameters:** `password` (str)

**Returns:** None (raises HTTPException if invalid)

**Side effects:** Raises 400 error if validation fails

**Used by:** `signup()`, `login()` endpoints

---

**`signup()`** — `backend/app/api/routes/auth.py`

**What it does:** Creates a new user account. Validates password, checks email uniqueness, hashes password, saves to DB, and returns JWT token immediately (direct login).

**Parameters:** 
- `name` (str) — Full name
- `email` (str) — Email address (must be unique)
- `password` (str) — Password (validated)
- `db` (AIOEngine) — Database (dependency)

**Returns:** 
```json
{
  "access_token": "eyJ0eX...",
  "token_type": "bearer",
  "user": {"name": "John", "email": "john@example.com", "created_at": "2025-03-31T..."}
}
```

**Side effects:** 
- Saves User record to MongoDB
- Sends welcome email (optional)
- Creates JWT token

**Used by:** Frontend `signup()` function in AuthContext

---

**`login()`** — `backend/app/api/routes/auth.py`

**What it does:** Authenticates user with email/password. Implements brute force protection (5 failed attempts = 5 min lockout). Updates last_login timestamp.

**Parameters:** 
- `email` (str)
- `password` (str)
- `db` (AIOEngine)

**Returns:** Access token + user info

**Side effects:** 
- Checks failed login attempts in memory
- Updates user.last_login in MongoDB
- Raises 429 if too many failed attempts

**Used by:** Frontend `login()` function

---

**`google_login()`** — `backend/app/api/routes/auth.py`

**What it does:** Authenticates user via Google OAuth token. Validates token with Google API, creates/finds user by email, returns JWT.

**Parameters:** 
- `token` (str) — Google access token
- `db` (AIOEngine)

**Returns:** Access token + user info

**Side effects:** 
- Calls Google userinfo API
- Creates/updates User in MongoDB
- Returns JWT

**Used by:** Frontend `googleLogin()` function

---

**`test_email_diag()`** — `backend/app/api/routes/auth.py`

**What it does:** Diagnostic endpoint to test email delivery. Sends a test OTP email and returns success/error details including traceback.

**Parameters:** `email` (str, query parameter)

**Returns:** 
```json
{
  "status": "success|error",
  "message": "Test email sent...",
  "error_type": "...",
  "error_detail": "...",
  "traceback": "..."
}
```

**Side effects:** Sends test email via Resend API

**Used by:** Developers debugging email issues

---

### `backend/app/api/routes/users.py` - User Management Routes

---

**`user_pulse()`** — `backend/app/api/routes/users.py`

**What it does:** Updates the user's "last seen" timestamp in ActiveSession. Called periodically (every 60 seconds) to track real-time active users.

**Parameters:** 
- `current_user` (User) — Current authenticated user (dependency)
- `db` (AIOEngine)

**Returns:** `{"status": "ok"}`

**Side effects:** Updates/creates ActiveSession record in MongoDB

**Used by:** Frontend every 60 seconds in useEffect hook (AuthContext)

---

**`read_users_me()`** — `backend/app/api/routes/users.py`

**What it does:** Returns the current authenticated user's complete profile data.

**Parameters:** `current_user` (User) — From JWT (dependency)

**Returns:** User object

**Side effects:** None (read-only)

**Used by:** Frontend to fetch current user info

---

**`change_password()`** — `backend/app/api/routes/users.py`

**What it does:** Changes user's password. Validates old password, enforces password policy on new password, updates DB.

**Parameters:** 
- `current_password` (str)
- `new_password` (str)
- `current_user` (User)
- `db` (AIOEngine)

**Returns:** `{"message": "Password updated successfully"}`

**Side effects:** 
- Validates old password
- Hashes new password
- Updates User.hashed_password in MongoDB

**Used by:** Frontend Profile page password change form

---

**`delete_account()`** — `backend/app/api/routes/users.py`

**What it does:** Permanently deletes user account and all associated scan data. One-way action.

**Parameters:** 
- `current_user` (User)
- `db` (AIOEngine)

**Returns:** `{"message": "Account and all associated data deleted successfully"}`

**Side effects:** 
- Deletes all SkinScan records for user
- Deletes User record
- Logs deletion

**Used by:** Frontend Profile page delete account button

---

### `backend/app/api/routes/scan.py` - Skin Disease Detection Routes

---

**`_upload_to_cloudinary_sync()`** — `backend/app/api/routes/scan.py`

**What it does:** Background task that uploads image to Cloudinary CDN and updates DB with final image URL. Runs asynchronously after prediction.

**Parameters:** 
- `image_bytes` (bytes) — Raw image binary data
- `scan_id` (str) — MongoDB scan record ID

**Returns:** None

**Side effects:** 
- Uploads to Cloudinary
- Updates SkinScan.image_url in MongoDB
- Logs warnings if upload fails

**Used by:** `predict_disease()` as background task

---

**`predict_disease()`** — `backend/app/api/routes/scan.py`

**What it does:** Main feature endpoint. Validates image file, runs ML inference, returns disease prediction with confidence/severity/recommendations.

**Parameters:** 
- `file` (UploadFile) — Image file from multipart form
- `background_tasks` (BackgroundTasks) — FastAPI background tasks

**Returns:** 
```json
{
  "id": "scan-id",
  "disease_detected": "Acne",
  "confidence_score": 0.92,
  "severity_level": "Mild",
  "description": "Common skin condition...",
  "recommendation": "Use gentle cleanser...",
  "do_list": [...],
  "dont_list": [...],
  "created_at": "2025-03-31T..."
}
```

**Side effects:** 
- Validates file (type, size, integrity)
- Runs ML model inference (CPU/GPU)
- Triggers background Cloudinary upload
- Note: Currently doesn't save to DB (commented out)

**Used by:** Frontend Detect page

---

**`get_scan_history()`** — `backend/app/api/routes/scan.py`

**What it does:** Retrieves all of user's past skin scans sorted by date (newest first).

**Parameters:** 
- `current_user` (User)
- `db` (AIOEngine)

**Returns:** Array of scan objects
```json
[
  {
    "id": "...",
    "disease_detected": "Acne",
    "confidence_score": 0.92,
    "severity_level": "Mild",
    "description": "...",
    "recommendation": "...",
    "created_at": "2025-03-31T...",
    "do_list": [...],
    "dont_list": [...]
  }
]
```

**Side effects:** Queries MongoDB for user's SkinScan records

**Used by:** Frontend Tracker.jsx, Profile.jsx pages

---

**`get_upload_limit()`** — `backend/app/api/routes/scan.py`

**What it does:** Returns remaining daily scans for the user. Limit is 5 per day, resets at midnight UTC.

**Parameters:** 
- `current_user` (User)
- `db` (AIOEngine)

**Returns:** 
```json
{
  "remaining": 3,
  "limit": 5
}
```

**Side effects:** Counts user's scans from today in MongoDB

**Used by:** Frontend Detect page to show remaining uploads

---

### `backend/app/api/routes/doctors.py` - Doctor Finder Routes

---

**`get_nearby_doctors()`** — `backend/app/api/routes/doctors.py`

**What it does:** Finds dermatologists near user's GPS coordinates using AWS Location Services API.

**Parameters:** `coords` (Coordinates object with latitude, longitude)

**Returns:** 
```json
[
  {
    "name": "Dr. Smith Dermatology Clinic",
    "address": "123 Main St, City",
    "rating": 4.5,
    "user_ratings_total": 120,
    "open_now": true,
    "place_id": "...",
    "geometry": {"location": {"lat": 40.7128, "lng": -74.0060}}
  }
]
```

**Side effects:** Calls AWS Location Services API

**Used by:** Frontend Doctors page DoctorFinder component

---

### `backend/app/api/routes/admin.py` - Admin Dashboard Routes

---

**`verify_admin()`** — `backend/app/api/routes/admin.py`

**What it does:** Validates admin password from X-Admin-Password header. Used as dependency for admin-only endpoints.

**Parameters:** `x_admin_password` (str, optional header)

**Returns:** True if valid

**Raises:** HTTPException 401 if invalid

**Side effects:** None

**Used by:** `get_admin_stats()` as dependency [SECURITY - weak protection, uses hardcoded password]

---

**`get_admin_stats()`** — `backend/app/api/routes/admin.py`

**What it does:** Returns system-wide statistics including total users, active users, live users, and recent activity.

**Parameters:** Requires admin password in header (dependency)

**Returns:** 
```json
{
  "total_users": 150,
  "active_users_24h": 45,
  "live_users": 8,
  "total_scans": 450,
  "recent_users": [...],
  "recent_scans": [...]
}
```

**Side effects:** 
- Queries User count
- Queries ActiveSession for live users
- Queries SkinScan count

**Used by:** Frontend Admin.jsx page

---

### `backend/app/services/ml_model.py` - ML Inference Service

---

**`_get_model()`** — `backend/app/services/ml_model.py`

**What it does:** Loads the pre-trained MobileNetV2 model from HDF5 file. Caches in global `_model` variable for reuse.

**Parameters:** None

**Returns:** Tensorflow/Keras model instance

**Side effects:** 
- Loads HDF5 file from disk (~100MB)
- Sets global `_model` variable
- Logs status

**Used by:** `predict()` function, `startup_event()` in main.py

---

**`predict()`** — `backend/app/services/ml_model.py`

**What it does:** Main ML inference function. Takes image bytes, preprocesses, runs model prediction, and returns disease with confidence.

**Parameters:** `image_bytes` (bytes) — Raw image binary data

**Returns:** 
```python
{
    "disease": "Acne",
    "category": "Acne",
    "confidence": 0.92,
    "severity": "Mild",
    "description": "A common skin condition...",
    "recommendation": "Use gentle cleanser...",
    "do_list": [...],
    "dont_list": [...],
    "is_unknown": False,
    "includes": []
}
```

**Side effects:** 
- Decodes image bytes
- Resizes to 224x224
- Normalizes pixel values
- Runs expensive ML computation (CPU/GPU)

**Used by:** `predict_disease()` route

---

### `backend/app/services/email.py` - Email Service

---

**`send_otp_email()`** — `backend/app/services/email.py`

**What it does:** Sends OTP verification email via Resend API. Email contains 6-digit code that expires in 10 minutes.

**Parameters:** 
- `email` (EmailStr) — Recipient email
- `otp` (str) — 6-digit OTP code

**Returns:** None (async)

**Side effects:** 
- Calls Resend API
- Sends email with HTML template
- Logs success/error

**Used by:** OTP-related endpoints (if fully implemented)

---

**`send_welcome_email()`** — `backend/app/services/email.py`

**What it does:** Sends welcome email to new users after signup.

**Parameters:** 
- `email` (EmailStr)
- `name` (str) — User's first name

**Returns:** None (async)

**Side effects:** Calls Resend API

**Used by:** `signup()` endpoint

---

### `backend/app/services/maps.py` - Location Service

---

**`find_nearby_dermatologists()`** — `backend/app/services/maps.py`

**What it does:** Searches for nearby dermatologists using AWS Location Services (Geo Places API v2). Returns list of doctors with details.

**Parameters:** 
- `lat` (float) — Latitude
- `lng` (float) — Longitude
- `radius` (int, default 5000) — Search radius in meters

**Returns:** 
```python
[
    {
        "name": "Dr. Smith",
        "address": "123 Main St",
        "rating": None,
        "user_ratings_total": 0,
        "open_now": None,
        "place_id": "...",
        "geometry": {"location": {"lat": 40.7128, "lng": -74.0060}}
    }
]
```

**Side effects:** 
- Calls AWS Location API
- Returns mock data if API key not set

**Used by:** `get_nearby_doctors()` route

---

---

## Frontend Functions

### `frontend/src/main.jsx` - Entry Point

(No exported functions - setup code only)

---

### `frontend/src/context/AuthContext.jsx` - Authentication Context

---

**`isTokenExpired()`** — `frontend/src/context/AuthContext.jsx`

**What it does:** Decodes JWT token and checks if it's expired by comparing exp claim to current time.

**Parameters:** `token` (str) — JWT token

**Returns:** Boolean (true if expired, false if valid)

**Side effects:** None

**Used by:** AuthProvider on component mount

---

**`useAuth()`** — `frontend/src/context/AuthContext.jsx`

**What it does:** React hook to access authentication context. Returns user state and auth methods.

**Parameters:** None

**Returns:** 
```javascript
{
  user: {name, email, created_at} | null,
  login: async function,
  signup: async function,
  googleLogin: async function,
  logout: function,
  verifyOtp: async function,
  loading: boolean
}
```

**Side effects:** None (hook)

**Used by:** Every authenticated page/component (Detect, Tracker, Profile, Doctors)

---

**`AuthProvider.login()`** — `frontend/src/context/AuthContext.jsx`

**What it does:** Calls backend `/api/v1/auth/login` endpoint with email/password, stores JWT + user info in localStorage.

**Parameters:** 
- `email` (str)
- `password` (str)

**Returns:** 
```javascript
{
  success: true/false,
  error: "error message" | undefined
}
```

**Side effects:** 
- Calls backend API
- Sets localStorage.token
- Sets localStorage.user
- Updates React state

**Used by:** Login.jsx form submission

---

**`AuthProvider.signup()`** — `frontend/src/context/AuthContext.jsx`

**What it does:** Calls backend `/api/v1/auth/signup` endpoint, handles direct login (immediate token) or returns success.

**Parameters:** 
- `name` (str)
- `email` (str)
- `password` (str)

**Returns:** 
```javascript
{
  success: true/false,
  error: "error message",
  user: {name, email, created_at} | undefined
}
```

**Side effects:** API call, localStorage updates, state changes

**Used by:** Signup.jsx form

---

**`AuthProvider.googleLogin()`** — `frontend/src/context/AuthContext.jsx`

**What it does:** Calls backend `/api/v1/auth/google` endpoint with Google access token, authenticates user.

**Parameters:** `token` (str) — Google access token from @react-oauth/google

**Returns:** `{success, error}`

**Side effects:** API call, localStorage updates

**Used by:** Login.jsx and Signup.jsx Google login button

---

**`AuthProvider.logout()`** — `frontend/src/context/AuthContext.jsx`

**What it does:** Clears authentication state. Removes JWT token and user from localStorage, sets user to null.

**Parameters:** None

**Returns:** None

**Side effects:** 
- Clears localStorage.token
- Clears localStorage.user
- Sets user state to null

**Used by:** Navbar logout button, Profile.jsx logout

---

### `frontend/src/pages/Detect.jsx` - AI Detection Page

---

**`compressImage()`** — `frontend/src/pages/Detect.jsx`

**What it does:** Compresses image before upload using Canvas API. Reduces size from ~5MB to ~100-200KB while maintaining acceptable quality.

**Parameters:** 
- `imageFile` (File) — Selected image file
- `maxSize` (number, default 800) — Max width/height in pixels
- `quality` (number, default 0.8) — JPEG quality (0-1)

**Returns:** Promise<File> — Compressed image file

**Side effects:** Creates Canvas element, draws/compresses image

**Used by:** `handleAnalyze()` before upload

---

**`handleAnalyze()`** — `frontend/src/pages/Detect.jsx`

**What it does:** Main analysis flow. Compresses image, uploads to backend, shows loading animation, displays results.

**Parameters:** None (uses component state)

**Returns:** None

**Side effects:** 
- Calls `compressImage()`
- Calls `/api/v1/scan/predict` with FormData
- Updates isAnalyzing, result, error state

**Used by:** "Analyze" button click

---

**`handleDownloadPDF()`** — `frontend/src/pages/Detect.jsx`

**What it does:** Triggers PDF download of medical report for current scan result.

**Parameters:** None

**Returns:** None

**Side effects:** Calls `downloadMedicalReport()` utility

**Used by:** PDF download button in results

---

### `frontend/src/pages/Tracker.jsx` - Scan History Page

---

**`fetchHistory()`** — `frontend/src/pages/Tracker.jsx`

**What it does:** Calls backend `/api/v1/scan/history` endpoint to fetch all user's past scans.

**Parameters:** None

**Returns:** None (sets history state)

**Side effects:** 
- API call with JWT
- Updates history state
- Sets loading, error states

**Used by:** useEffect hook on component mount

---

### `frontend/src/pages/Profile.jsx` - User Profile Page

---

**`fetchHistory()`** — `frontend/src/pages/Profile.jsx`

**What it does:** Fetches user's scan history to display in profile page.

**Parameters:** None

**Returns:** None

**Side effects:** API call, state updates

**Used by:** useEffect hook

---

**`handleChangePassword()`** — `frontend/src/pages/Profile.jsx`

**What it does:** Calls backend `/api/v1/users/change-password` endpoint with old and new password.

**Parameters:** None (uses component form state)

**Returns:** None

**Side effects:** 
- Validates passwords match
- API call
- Updates changePasswordStatus state
- Clears form on success

**Used by:** Password change form submission

---

**`handleLogout()`** — `frontend/src/pages/Profile.jsx`

**What it does:** Calls logout from AuthContext and navigates to login page.

**Parameters:** None

**Returns:** None

**Side effects:** 
- Calls AuthProvider.logout()
- Navigates via react-router

**Used by:** Logout button

---

### `frontend/src/utils/pdfGenerator.js` - PDF Report Generation

---

**`downloadMedicalReport()`** — `frontend/src/utils/pdfGenerator.js`

**What it does:** Generates and downloads a professional medical report PDF for a skin scan result using jsPDF.

**Parameters:** 
- `result` (object) — Scan result {disease, confidence, severity, description, recommendation, doList, dontList}
- `user` (object) — Current user {name, email}

**Returns:** None (triggers browser download)

**Side effects:** 
- Creates PDF document
- Triggers browser download
- No database changes

**Used by:** Download PDF button in Detect.jsx and Profile.jsx

---

### `frontend/src/utils/cn.js` - CSS Utility

---

**`cn()`** — `frontend/src/utils/cn.js`

**What it does:** Helper function to merge Tailwind CSS classes intelligently, handling conflicts and duplicates.

**Parameters:** `...inputs` (variable arguments) — CSS class strings or arrays

**Returns:** String of merged CSS classes

**Side effects:** None

**Used by:** Throughout component files for dynamic className generation

---

### `frontend/src/components/DoctorFinder.jsx` - Doctor Search Component

---

**`DoctorFinder`** — `frontend/src/components/DoctorFinder.jsx` (Component, not function)

**What it does:** React component that gets user's GPS location and searches for nearby dermatologists using `/api/v1/doctors/nearby` endpoint.

**Props:** None

**Returns:** JSX (doctor list with address, rating, location)

**Side effects:** 
- Requests browser geolocation permission
- Calls backend API with coordinates
- Updates component state

**Used by:** Doctors.jsx page

---

### `frontend/src/components/DayNightToggle.jsx` - Theme Toggle

---

**`DayNightToggle`** — `frontend/src/components/DayNightToggle.jsx` (Component)

**What it does:** React component that toggles between light/dark theme by setting CSS variables.

**Props:** None

**Returns:** JSX (toggle button)

**Side effects:** Updates CSS variables on document root element

**Used by:** Navbar.jsx

---

---

## Summary Statistics

| Category | Count |
|---|---|
| Backend API Routes | 20+ |
| Backend Service Functions | 8 |
| Core Auth/DB Functions | 5 |
| Frontend Context/State Functions | 6 |
| Frontend Page Functions | 8+ |
| Frontend Component Functions | 2 |
| Utility Functions | 2 |
| **Total Functions Documented** | **50+** |

---

## Function Call Graph (Key Dependencies)

```
Frontend Entry Point (main.jsx)
├── App.jsx
│   ├── Login.jsx → AuthContext.login()
│   ├── Signup.jsx → AuthContext.signup()
│   ├── Detect.jsx
│   │   ├── compressImage()
│   │   └── handleAnalyze() → /api/v1/scan/predict
│   ├── Tracker.jsx → fetchHistory() → /api/v1/scan/history
│   ├── Profile.jsx
│   │   ├── fetchHistory()
│   │   ├── handleChangePassword() → /api/v1/users/change-password
│   │   └── handleLogout() → AuthContext.logout()
│   └── Doctors.jsx → DoctorFinder → /api/v1/doctors/nearby
│
Backend Entry Point (main.py)
├── startup_event() → _get_model()
├── /auth routes
│   ├── signup() → verify_password(), get_password_hash(), create_access_token()
│   ├── login() → verify_password(), create_access_token()
│   └── google_login() → create_access_token()
├── /users routes
│   ├── user_pulse() → db update
│   ├── change_password() → verify_password(), get_password_hash()
│   └── delete_account() → db delete
├── /scan routes
│   ├── predict_disease() → ml_predict(), _upload_to_cloudinary_sync()
│   ├── get_scan_history() → db query
│   └── get_upload_limit() → db count
├── /doctors routes
│   └── get_nearby_doctors() → find_nearby_dermatologists()
└── /admin routes
    └── get_admin_stats() → db queries
```

