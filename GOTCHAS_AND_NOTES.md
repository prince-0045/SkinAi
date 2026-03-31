# GOTCHAS AND NOTES

Important gotchas, non-obvious behaviors, tricky parts, hardcoded values, known issues, and critical information for developers working on this codebase.

---

## Critical Reading Before You Start

### 1. Scan Data NOT Fully Saved to Database

**Issue:** The scan prediction endpoint returns results, but DOES NOT save them to MongoDB (code is commented out).

**Location:** `backend/app/api/routes/scan.py` → `predict_disease()` lines ~90-110

**Current State:**
```python
# ── Save to DB with placeholder URL ──
# scan = SkinScan(...)
# await db.save(scan)

# ── Upload to Cloudinary in background ──
# background_tasks.add_task(_upload_to_cloudinary_sync, image_bytes, str(scan.id))

# Currently just returns test data
return {
    "id": "test-scan-id",
    "user_id": "test-user",
    ...
}
```

**Impact:**
- Scan results are never persisted
- Tracker page shows empty history (no fetched data)
- Every refresh loses the analysis
- PDF generation only works with in-memory data

**Solution (To Fix):**
1. Un-comment the scan creation code
2. Un-comment the background task
3. Fix the test data to use real data
4. Test end-to-end flow

**Priority:** 🔴 CRITICAL - Must fix for app to be usable

---

### 2. Admin Endpoint Has NO Proper Authentication

**Issue:** Admin stats endpoint uses hardcoded password in header, not JWT or OAuth.

**Location:** `backend/app/api/routes/admin.py` → `verify_admin()`

**Current Implementation:**
```python
def verify_admin(x_admin_password: Optional[str] = Header(None)):
    if not x_admin_password or x_admin_password != settings.ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True
```

**Problem:**
- Password sent in plain headers (should use JWT)
- Password stored in environment (security risk)
- No rate limiting on password attempts
- Frontend sends password from JavaScript (visible in DevTools)

**Recommendation:** Change to JWT-based admin role

---

### 3. User Authentication Can Bypass Password

**Issue:** Backend auth accepts plain text passwords during "bypass mode" for legacy data.

**Location:** `backend/app/core/auth.py` → `verify_password()`

```python
def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        # Fallback for plain text passwords
        print(f"DEBUG: verify_password fallback check for legacy user: {e}")
        return plain_password == hashed_password  # ← DANGEROUS
```

**Impact:**
- If password is stored as plain text, it accepts exact match
- Reduces security for legacy users

**Status:** This is a debug/migration feature, should be removed once all passwords are hashed

---

### 4. OTP Email Verification Not Implemented

**Issue:** OTP system has frontend and backend endpoints but they're not connected.

**Locations:**
- Frontend: `frontend/src/pages/OTP.jsx` (incomplete)
- Backend: `backend/app/api/routes/auth.py` → `test_email_diag()` (diagnostic only)
- Endpoint: POST `/auth/verify-otp` (doesn't exist)

**Current Flow:**
```
User signup → Should send OTP → OTP page → Verify → Login
Currently:
User signup → Direct login (is_verified defaults to True)
```

**Note:** Current signup defaults `is_verified=True`, so OTP is bypassed. Verification code in `/users/me` is commented out:

```python
# if not user.is_verified:
#     raise HTTPException(...)
```

**To Implement:**
1. Create POST `/auth/verify-otp` endpoint
2. Add OTP model to domain.py
3. Save OTP in MongoDB with expiry
4. Check OTP in verify endpoint
5. Complete OTP.jsx frontend

---

### 5. ML Model Lazy Loading

**Issue:** ML model isn't required on startup but tries to load anyway.

**Location:** `backend/app/main.py` → `startup_event()`

```python
try:
    from app.services.ml_model import _get_model
    _get_model()  # Pre-loads on startup
except Exception as e:
    logger.warning(f"Model pre-load failed (will lazy-load): {e}")
```

**Behavior:**
- Tries to load model on startup
- If fails, logs warning and continues
- Model loads on first `/scan/predict` request (lazy loading)

**Performance Impact:**
- Startup takes ~10-30 seconds (model is 100MB+)
- First scan request is slow if model not pre-loaded
- Subsequent scans are faster (cached)

**Optimization Options:**
- Remove pre-load (faster startup, slower first request)
- Load in background thread (non-blocking startup)
- Use model quantization (smaller model size)

---

### 6. Daily Scan Limit Disabled

**Issue:** Daily scan limit check is commented out.

**Location:** `backend/app/api/routes/scan.py` → `predict_disease()`

```python
# ── Check daily upload limit ──
# try:
#     daily_scans_count = await db.count(SkinScan, ...)
# except Exception:
#     daily_scans_count = 0
# 
# if daily_scans_count >= DAILY_SCAN_LIMIT:
#     raise HTTPException(429, "Daily upload limit reached...")
```

**Current State:**
- Can upload unlimited scans per day
- Endpoint `/scan/limit` still works (returns remaining limit)
- Constants defined: `DAILY_SCAN_LIMIT = 5`

**To Enable:**
1. Un-comment the check
2. Also un-comment DB saves (above issue)
3. Test rate limiting

---

### 7. Cloudinary Upload is Async, URL Update is Delayed

**Issue:** Image URL isn't available immediately after scan.

**Location:** `backend/app/api/routes/scan.py` → `_upload_to_cloudinary_sync()`

**Flow:**
```
1. User uploads image
2. Backend returns response with image_url: "pending"
3. Background task uploads to Cloudinary (async)
4. Updates DB with real CDN URL later
```

**Impact:**
- Immediate response has `image_url: "pending"`
- Real URL available after ~10 seconds
- Frontend should handle "pending" state
- Currently not saved to DB (so no actual update)

**Note:** Uses synchronous Cloudinary client inside async background task. This is blocking and should be refactored to use async HTTP library.

---

### 8. Hardcoded Google OAuth Client ID

**Issue:** Google client ID is hardcoded in JavaScript.

**Location:** `frontend/src/main.jsx`

```javascript
<GoogleOAuthProvider clientId="144232855654-v2bucqk1f3679d06gedppcj1sd3moqnd.apps.googleusercontent.com">
```

**Why It's OK:**
- Client IDs are meant to be public
- Frontend-only, secure for public use
- Backend validates tokens with Google API

**However:**
- Could be moved to environment variable for flexibility

---

### 9. Admin Password Stored in Environment

**Issue:** Admin password is plain text in `.env` file.

**Location:** `backend/app/core/config.py`

```python
ADMIN_PASSWORD: str = "admin123"  # User should change this in .env
```

**Security Concerns:**
- Default value is weak ("admin123")
- Stored in plain text (should be hashed)
- Environment variables visible in logs
- No rate limiting on attempts

**Recommendation:**
1. Change default to strong random string
2. Implement proper admin user model with hashed passwords
3. Use JWT-based admin authentication
4. Add rate limiting

---

### 10. ML Model Confidence Threshold Edge Case

**Issue:** When confidence score is below 35% (MIN_CONFIDENCE_THRESHOLD), disease is returned as "Unknown".

**Location:** 
- Constants: `backend/app/core/constants.py` → `MIN_CONFIDENCE_THRESHOLD = 0.35`
- Model: `backend/app/services/ml_model.py`

**Behavior:**
```python
if confidence < 0.35:
    disease = "Unknown"  # Instead of most confident class
```

**Impact:**
- Shows "Unknown" condition to user even if model has prediction
- Severity level, description, etc. for "Unknown" disease are generic
- User might think app failed instead of uncertain result

**Recommendation:**
- Show top prediction AND confidence score
- Let user decide if they trust the prediction
- Or increase threshold if too many false "Unknown" results

---

### 11. No Input Validation on Image Metadata

**Issue:** Backend validates file type and size but not image dimensions or aspect ratio.

**Location:** `backend/app/api/routes/scan.py` → `predict_disease()`

```python
# File type check
if file.content_type not in ALLOWED_IMAGE_TYPES:
    raise HTTPException(400, "Invalid file type")

# File size check
if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
    raise HTTPException(400, "File too large")

# Image integrity check (but no dimension check)
Image.open(BytesIO(image_bytes)).verify()

# Image is resized to 224x224 regardless of original aspect ratio
# Could distort images that aren't square
```

**Impact:**
- Model expects 224x224, input is resized (may be distorted)
- No check for blurry, too dark, or invalid images
- Could affect ML accuracy

**Improvement:**
1. Add dimension check (warn if not square)
2. Add brightness/contrast validation
3. Reject obviously invalid images

---

### 12. Brute Force Protection Uses In-Memory Dictionary

**Issue:** Failed login attempts tracked in memory, not persistent.

**Location:** `backend/app/api/routes/auth.py`

```python
_login_attempts = defaultdict(list)  # In-memory, lost on restart
```

**Problem:**
- Resets when server restarts
- Doesn't persist across multiple server instances (load balancing)
- No cleanup of old entries (memory leak potential)

**Recommendation:**
- Store in Redis for distributed systems
- Or use TimedDict with expiry

---

### 13. CORS Configuration Allows All Subdomains

**Issue:** CORS allows any `*.dermaura.tech` subdomain.

**Location:** `backend/app/main.py`

```python
allow_origin_regex=r"https?://(.*\.)?dermaura\.tech"
```

**Impact:**
- Any attacker can create subdomain and CORS will allow it
- If DNS is compromised, API can be accessed from anywhere

**Recommendation:**
- Whitelist specific known subdomains
- Or use environment variable for allowed origins

---

### 14. Database Connection Pooling Tight

**Issue:** Connection pool might be too small or too large.

**Location:** `backend/app/core/database.py`

```python
self.client = AsyncIOMotorClient(
    db_url,
    maxPoolSize=50,
    minPoolSize=10,
    ...
)
```

**Note:** These values are hardcoded. Should be environment variables for different deployments.

**Tuning:**
- Production: May need maxPoolSize > 50
- Local dev: minPoolSize=10 is wasteful
- Adjust based on load testing

---

### 15. Frontend API_BASE_URL Can Be Empty

**Issue:** API_BASE_URL defaults to empty string if env var not set.

**Location:** `frontend/src/context/AuthContext.jsx` and throughout pages

```javascript
const API_BASE_URL = import.meta.env.VITE_API_URL || '';
// Empty string causes: fetch('http://localhost/api/v1/auth/login')
```

**Problem:**
- Relative URLs might not work as expected
- Could cause CORS issues
- Silently fails in production if env not set

**Recommendation:**
- Define default for production: `https://dermaura.tech`
- Throw error if missing instead of defaulting to empty

---

### 16. No Request Timeout on Long-Running Scans

**Issue:** ML inference can take 30+ seconds, but frontend has no timeout mechanism.

**Impact:**
- Long inference → User thinks app is broken
- No visual feedback after initial "Analyzing..." spinner
- Could improve UX with progress indication

---

### 17. Scan History Not Filtered by Date Range

**Issue:** `/scan/history` returns ALL scans, no pagination or date filtering.

**Location:** `backend/app/api/routes/scan.py` → `get_scan_history()`

**Impact:**
- If user has 10,000 scans, all loaded at once
- Frontend chart might become slow
- No way to view "scans from last month" etc.

**Recommendation:**
- Add pagination (limit, offset)
- Add date range filtering
- Sort options

---

### 18. Email Service Silently Fails in Some Cases

**Issue:** Email errors are logged but not returned to user in some endpoints.

**Location:** `backend/app/services/email.py`

```python
except Exception as e:
    logger.error(f"Error sending welcome email: {e}")
    # No re-raise, just log
```

**Impact:**
- User thinks email was sent but it wasn't
- Welcome email goes missing silently
- OTP emails might also fail silently

**Recommendation:**
- Log at different levels (warning, error)
- Let caller decide whether to propagate

---

### 19. No Offline Support

**Issue:** App requires constant internet connection.

**Impact:**
- Can't use app in offline mode
- No service worker or offline caching
- User loses work if connection drops mid-upload

**Note:** Vite PWA plugin is installed (`vite-plugin-pwa`) but not configured.

---

### 20. ML Model Uses CPU-Only

**Issue:** TensorFlow is CPU version (`tensorflow-cpu`), not GPU.

**Location:** `backend/requirements.txt`

```
tensorflow-cpu
```

**Impact:**
- Inference is slow (30+ seconds)
- Large server costs for compute
- Could use GPU for production

**Recommendation:**
- Use `tensorflow` or `tensorflow-gpu` for production
- Or use quantized model for faster inference

---

## Magic Numbers & Constants

| Value | Location | Meaning |
|---|---|---|
| 224x224 | `services/ml_model.py` | ML model input size |
| 0.35 | `constants.py` | Confidence threshold for "Unknown" |
| 13 | `services/ml_model.py` | Number of disease classes |
| 5 | `constants.py` | Daily scan limit |
| 10MB | `constants.py` | Max upload file size |
| 8 | `constants.py` | Min password length |
| 5 | `constants.py` | Max failed login attempts |
| 300 | `constants.py` | Login lockout seconds (5 minutes) |
| 10 | `constants.py` | OTP expiry minutes |
| 24 | `constants.py` | JWT expiry hours |
| 1000/minute | `main.py` | Global rate limit |
| 800x800 | `pages/Detect.jsx` | Image compression size |
| 0.8 | `pages/Detect.jsx` | JPEG compression quality |
| 60000ms | `context/AuthContext.jsx` | Pulse interval (1 minute) |
| 2 | `routes/admin.py` | Live user threshold (minutes) |
| 10 | `routes/admin.py` | Recent scans to show (limit) |

---

## Common Mistakes Developers Make

### 1. Forgetting to Add JWT Token to Requests

```javascript
// ❌ Wrong - No authorization
const response = await fetch('/api/v1/scan/predict', { ... });

// ✅ Correct
const token = localStorage.getItem('token');
const response = await fetch('/api/v1/scan/predict', {
  headers: { Authorization: `Bearer ${token}` }
});
```

### 2. Assuming Scan is Saved Immediately

```javascript
// ❌ Wrong - Scan isn't saved to DB currently
const result = predict_disease();
// result is not in scan history yet

// ✓ Correct (after fixing DB save)
const result = predict_disease();
// Wait for /scan/history to include new scan
```

### 3. Not Handling Loading State

```javascript
// ❌ Wrong - Flash of confusion
<button onClick={handleUpload}>Upload</button>

// ✓ Correct
<button onClick={handleUpload} disabled={isAnalyzing}>
  {isAnalyzing ? 'Analyzing...' : 'Upload'}
</button>
```

### 4. Not Checking User in Protected Routes

```javascript
// ❌ Wrong - Page might show before auth checks
export default function Profile() {
  const { user } = useAuth();
  return <div>{user.name}</div>; // Could crash if user is null
}

// ✓ Correct
export default function Profile() {
  const { user, loading } = useAuth();
  if (loading) return <Loading />;
  if (!user) return <NotLoggedIn />;
  return <div>{user.name}</div>;
}
```

### 5. Modifying Constants Without Thinking Through Impact

```python
# Changing MIN_CONFIDENCE_THRESHOLD from 0.35 to 0.9
# Impact: Almost everything becomes "Unknown"
# Better: Test with different values first
```

---

## Testing Checklist

Before pushing changes:

- [ ] Signup works end-to-end
- [ ] Login works with email/password
- [ ] Google Sign-In works
- [ ] Scan upload and analysis completes
- [ ] Scan results display correctly
- [ ] Scan history shows recent scans
- [ ] Profile page loads user info
- [ ] Password change works
- [ ] Account deletion works
- [ ] Doctor finder returns nearby doctors
- [ ] Logout clears auth state
- [ ] Protected routes redirect unauthed users
- [ ] Admin stats accessible with password
- [ ] No console errors or warnings
- [ ] Mobile responsive (test on small screen)
- [ ] Dark theme toggles properly
- [ ] API rate limiting activates at 1000 req/min

---

## Performance Gotchas

1. **ML Inference is Slow** - First scan takes 30+ seconds
2. **Large Image Upload** - 10MB max, but compression recommended
3. **Database Count Queries** - Counting all scans is slow with large dataset
4. **Chart Rendering** - Recharts with 1000+ points becomes sluggish
5. **No Caching** - Every page refresh hits API

---

## Security Gotchas

1. **Passwords in Error Messages** - Don't log plain passwords
2. **Token Expiry Not Handled** - User can get 401 mid-session
3. **No Rate Limiting on Password Reset** - Could enumerate emails
4. **CSS Variables in Browser** - Theme colors are public knowledge
5. **Google Client ID Public** - It's meant to be, but don't worry

---

## Browser Compatibility Notes

- Requires modern browser (ES2020+)
- localStorage required (no IE support)
- Geolocation API required for doctor finder
- Canvas API required for image compression
- Fetch API required (or fetch polyfill)

---

## Environment Variables Checklist

Backend `.env` must have:
```
PROJECT_NAME=SkinCare AI
MONGO_URL=mongodb+srv://...
SECRET_KEY=<random 32+ chars>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
MAIL_USERNAME=...
MAIL_PASSWORD=...
MAIL_FROM=...
MAIL_PORT=...
MAIL_SERVER=...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
ADMIN_PASSWORD=<strong password>
AWS_ACCESS_KEY_ID=... (optional)
AWS_SECRET_ACCESS_KEY=... (optional)
AWS_MAPS_API_KEY=... (optional)
RESEND_API_KEY=...
FRONTEND_URL=...
```

Frontend `.env.local` must have:
```
VITE_API_URL=http://localhost:8000 (or production URL)
```

---

## Debugging Tips

### Check Server Logs
```bash
tail -f server.log | grep ERROR
```

### Check ML Model Load
```bash
curl http://localhost:8000/health
# Should show ml_model: loaded
```

### Test API Directly
```bash
curl -X POST http://localhost:8000/api/v1/scan/predict \
  -H "Authorization: Bearer <token>" \
  -F "file=@image.jpg"
```

### Check Auth Token
```javascript
// In browser console
JSON.parse(atob(localStorage.token.split('.')[1]))
```

### Clear State
```javascript
localStorage.clear()
location.reload()
```

---

