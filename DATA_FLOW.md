# DATA FLOW

Complete trace of data movement through the SkinAI application. Shows how information flows from user actions through the frontend, backend, database, and back to the UI.

---

## 1. User Registration Flow

```
User fills signup form
    ↓
1. React captures form input (name, email, password)
   - Frontend: src/pages/Signup.jsx
   - State: {name, email, password}
    ↓
2. User clicks "Sign Up" button
   - onClick → handleSignup()
   - Validates password locally (8+ chars, uppercase, number)
    ↓
3. Frontend POST to /api/v1/auth/signup
   - Headers: Content-Type: application/json
   - Body: {name, email, password}
   - Code: AuthContext.signup()
    ↓
4. Backend receives request
   - Route: backend/app/api/routes/auth.py → signup()
   - Validates incoming data (Pydantic)
    ↓
5. DATABASE: Check if email exists
   - Query: db.find_one(User, User.email == email)
   - MongoDB collection: "users"
   - If exists & verified: Return 400 error
   - If exists & unverified: Delete old record
    ↓
6. Hash password (security operation)
   - Function: core/auth.py → get_password_hash(password)
   - Algorithm: Argon2 (slow hashing)
   - Result: Hashed string with salt
    ↓
7. Create User object
   - user = User(
       name=name,
       email=email,
       hashed_password=hashed_hash,
       auth_provider="email",
       is_verified=True,
       created_at=datetime.utcnow()
     )
    ↓
8. DATABASE: Save User record
   - Insert: db.save(new_user)
   - Collection: "users"
   - Fields stored: id, name, email, hashed_password, auth_provider, is_verified, created_at, last_login
    ↓
9. Generate JWT token
   - Function: core/auth.py → create_access_token()
   - Data encoded: {"sub": user.email, "exp": now + 24 hours}
   - Algorithm: HS256
   - Result: Signed JWT string
    ↓
10. Send welcome email (async, optional)
    - Function: services/email.py → send_welcome_email()
    - Via: Resend API
    - Template: HTML email with welcome message
    ↓
11. Return response to frontend
    - Status: 201 Created
    - Body: {
        access_token: "eyJ0eX...",
        token_type: "bearer",
        user: {name, email, created_at}
      }
    ↓
12. Frontend stores authentication data
    - localStorage.setItem('token', data.access_token)
    - localStorage.setItem('user', JSON.stringify(data.user))
    - AuthContext state: user = {name, email, created_at}
    ↓
13. Frontend navigates
    - navigate('/detect')
    - User can now access protected routes
    ↓
✓ User is now registered and logged in
```

---

## 2. Skin Disease Detection Flow

```
User goes to /detect page
    ↓
1. Frontend loads Detect.jsx component
   - Fetches upload limit: GET /api/v1/scan/limit
   - Shows drag-drop zone for image
    ↓
2. User selects/drops image file
   - onDrop callback triggered
   - File state: {name, size, type}
   - Preview URL created: URL.createObjectURL(file)
    ↓
3. User clicks "Analyze" button
   - handleAnalyze() function called
   - setIsAnalyzing(true) → Shows loading spinner
    ↓
4. Frontend compresses image
   - Function: Detect.jsx → compressImage()
   - Creates Canvas element
   - Draws image scaled to 800x800
   - Exports as JPEG (80% quality)
   - Result: Compressed File object (~100-200KB)
    ↓
5. Frontend sends to backend
   - Method: POST /api/v1/scan/predict
   - Headers:
     Authorization: Bearer <JWT_token>
   - Body: FormData with file + compressed image
   - Via: fetch() or axios
    ↓
6. Backend receives image
   - Route: backend/app/api/routes/scan.py → predict_disease()
   - Validates incoming JWT (get_current_user dependency)
    ↓
7. Backend validates image
   - Checks file.content_type in ALLOWED_IMAGE_TYPES
   - Checks file size (max 10MB)
   - Validates image integrity: PIL.Image.open() and .verify()
   - If invalid: Return 400 error
    ↓
8. Backend validates quota
   - Queries: db.count(SkinScan)
   - Checks today's scan count for user
   - Compares against DAILY_SCAN_LIMIT (5 scans/day)
   - [Currently disabled/commented out]
    ↓
9. ML Model Inference (expensive operation)
   - Function: services/ml_model.py → predict(image_bytes)
   
   Step 9a: Load model (or use cached version)
     - _get_model() loads model.weights.h5
     - MobileNetV2 pre-trained on skin diseases
     - Cached in global _model variable
   
   Step 9b: Preprocess image
     - image_bytes → PIL Image
     - Resize to 224x224 pixels
     - Convert to numpy array
     - Normalize pixel values (0-1 range)
   
   Step 9c: Run inference
     - model.predict(preprocessed_image)
     - Output: Probability distribution over 13 classes
     - Classes: [Acne, Actinic_Keratosis, Benign_Growth, ...]
   
   Step 9d: Post-process results
     - Find class with highest probability → disease_name
     - Get confidence score (probability of top class)
     - If confidence < MIN_CONFIDENCE_THRESHOLD (0.35) → Disease = "Unknown"
     - Lookup DISEASE_INFO[disease] → Get severity, description, recommendations
    ↓
10. Backend prepares response
    - Result object: {
        id: scan_id,
        disease_detected: "Acne",
        confidence_score: 0.92,
        severity_level: "Mild",
        description: "Common skin condition...",
        recommendation: "Use gentle cleanser...",
        do_list: [...],
        dont_list: [...],
        created_at: datetime.utcnow()
      }
    ↓
11. Database saves scan (commented out - needs to be re-enabled)
    - CREATE: SkinScan(
        user_id=user.id,
        image_url="pending",  # Will be updated by background task
        disease_detected=disease,
        confidence_score=confidence,
        severity_level=severity,
        description=description,
        recommendation=recommendation,
        do_list=do_list,
        dont_list=dont_list,
        created_at=datetime.utcnow()
      )
    - db.save(scan)
    - MongoDB collection: "skin_scans"
    ↓
12. Background task: Upload to Cloudinary
    - Function: scan.py → _upload_to_cloudinary_sync()
    - Runs asynchronously (doesn't block response)
    - Uploads image bytes to Cloudinary CDN
    - Gets back secure_url
    - Updates SkinScan.image_url with CDN URL
    - MongoDB: UPDATE skin_scans SET image_url = secure_url WHERE _id = scan_id
    ↓
13. Backend returns response
    - Status: 200 OK
    - Body: Result object (JSON)
    ↓
14. Frontend receives response
    - setIsAnalyzing(false)
    - setResult(data)
    - Triggers re-render
    ↓
15. Frontend displays results
    - Shows severity color (Low: blue, Moderate: darker blue, High: cyan, Critical: bright)
    - Shows confidence (as percentage)
    - Shows disease name prominently
    - Shows description, recommendations, do/don't lists
    - Offers download PDF button
    ↓
16. User downloads PDF report (optional)
    - Function: utils/pdfGenerator.js → downloadMedicalReport()
    - Uses jsPDF library
    - Generates professional medical report
    - Includes: patient name, email, date, results, recommendations
    - Browser downloads: DERMAURA_Report_YYYY-MM-DD.pdf
    ↓
✓ Scan analysis complete
```

---

## 3. View Scan History Flow

```
User navigates to /track page
    ↓
1. Frontend loads Tracker.jsx component
   - useEffect hook triggers on mount
    ↓
2. Frontend requests scan history
   - GET /api/v1/scan/history
   - Headers: Authorization: Bearer <JWT>
   - No body
    ↓
3. Backend processes request
   - Route: scan.py → get_scan_history()
   - Validates JWT → get_current_user()
   - Extracts user.id from token
    ↓
4. Database query
   - Query: db.find(SkinScan, SkinScan.user_id == str(current_user.id))
   - Sort by: created_at descending (newest first)
   - MongoDB collection: "skin_scans"
   - Returns: [scan1, scan2, scan3, ...]
    ↓
5. Backend transforms data
   - Converts each SkinScan to JSON
   - Formats timestamps
   - Includes all fields: disease, confidence, severity, description, recommendations, do_list, dont_list
    ↓
6. Backend returns response
   - Status: 200 OK
   - Body: Array of scan objects
    ↓
7. Frontend receives history array
   - setHistory(data)
   - Re-renders component
    ↓
8. Frontend displays history (user chooses view)
   
   Option A: Grid View
   ────────────────
   - Maps over history array
   - Creates ScanCard component for each scan
   - Shows: thumbnail, disease, confidence, severity, date
   - User can click card to see details
   
   Option B: Chart View
   ────────────────
   - Processes history into format for Recharts
   - Creates LineChart showing confidence trend over time
   - X-axis: Scan number (1, 2, 3, ...)
   - Y-axis: Confidence score (0-100%)
   - Each point is a scan with hover detail
    ↓
9. Frontend calculates summary stats
    - Total scans: history.length
    - Avg confidence: sum(confidence_score) / count
    - Unique conditions: Set of disease_detected values
   - Displays stats cards
    ↓
10. User interactions
    - Click scan card → Shows full details
    - Download PDF → Calls downloadMedicalReport()
    - Change view → Toggle between grid/chart
    ↓
✓ User can see complete scan history
```

---

## 4. User Authentication (Heartbeat) Flow

```
User is logged in (any authenticated page)
    ↓
1. AuthContext mounts
   - Checks localStorage for token
   - Validates token not expired
   - Sets user state
    ↓
2. useEffect hook for pulse
   - Triggers when user is set
   - Interval: 60 seconds
    ↓
3. Every 60 seconds (while logged in)
   - Frontend: POST /api/v1/users/pulse
   - Headers: Authorization: Bearer <JWT>
   - No body
    ↓
4. Backend processes heartbeat
   - Route: users.py → user_pulse()
   - Validates JWT → get_current_user()
   - Extracts user.id
    ↓
5. Database update
   - Query: db.find_one(ActiveSession, ActiveSession.user_id == str(user.id))
   - If exists:
     - UPDATE: session.last_seen_at = datetime.utcnow()
     - db.save(session)
   - If not exists:
     - CREATE: ActiveSession(user_id=str(user.id), last_seen_at=now)
     - db.save(new_session)
   - MongoDB collection: "active_sessions"
    ↓
6. Backend returns response
   - Status: 200 OK
   - Body: {"status": "ok"}
    ↓
7. Frontend silently receives response
   - No UI update
   - Error handling: silently ignores failures
    ↓
✓ User session remains active in system
```

---

## 5. Find Doctors Flow

```
User navigates to /doctors page
    ↓
1. Frontend loads DoctorFinder component
   - Requests browser geolocation permission
    ↓
2. Browser geolocation API
   - navigator.geolocation.getCurrentPosition()
   - Returns: {latitude, longitude, accuracy}
   - User may deny: Show error message
    ↓
3. Frontend sends to backend
   - POST /api/v1/doctors/nearby
   - Body: {latitude: 40.7128, longitude: -74.0060}
    ↓
4. Backend processes request
   - Route: doctors.py → get_nearby_doctors()
   - Calls services/maps.py → find_nearby_dermatologists()
    ↓
5. Backend calls AWS Location API
   - Endpoint: https://places.geo.{region}.amazonaws.com/v2/search-text
   - Method: POST
   - Body: {
       QueryText: "dermatologist",
       BiasPosition: [longitude, latitude],
       MaxResults: 5,
       FilterCountries: ["IN"]
     }
   - Headers: API key authentication
    ↓
6. AWS Location API returns results
   - Returns: {ResultItems: [...]}
   - Each item: {Title, Address, PlaceId, Position: [lng, lat]}
    ↓
7. Backend transforms data
   - Maps AWS format to internal format
   - Each doctor: {name, address, rating, user_ratings_total, open_now, place_id, geometry}
    ↓
8. Backend returns response to frontend
   - Status: 200 OK
   - Body: Array of doctor objects
    ↓
9. Frontend receives doctors
   - setDoctors(data)
   - Re-renders list
    ↓
10. Frontend displays doctors
    - Shows doctor name, address, rating
    - Shows distance from user
    - User can click for directions (Google Maps link)
    - User can call (phone link)
    ↓
✓ User can see nearby dermatologists
```

---

## 6. User Login Flow

```
User not logged in → goes to /login page
    ↓
1. Login.jsx displays form
   - Email input field
   - Password input field (with show/hide toggle)
   - Submit button
   - Link to signup page
    ↓
2. User enters credentials + clicks Login
   - Frontend captures email, password
   - Sets loading = true
    ↓
3. Frontend calls AuthContext.login()
   - POST /api/v1/auth/login
   - Headers: Content-Type: application/json
   - Body: {email, password}
    ↓
4. Backend receives credentials
   - Route: auth.py → login()
   - Validates incoming data (Pydantic)
    ↓
5. Brute force protection check
   - Variables: _login_attempts = {email: [timestamp1, timestamp2, ...]}
   - Current time: now = time.time()
   - Filter: recent = [t for t in _login_attempts[email] if now - t < 300seconds]
   - If len(recent) >= 5:
     - Return 429 Too Many Requests
     - Detail: "Please try again in 5 minutes"
   - Else: Continue
    ↓
6. Database lookup
   - Query: db.find_one(User, User.email == email)
   - If not found:
     - Record failed attempt: _login_attempts[email].append(now)
     - Return 401 Unauthorized: "Invalid credentials"
    ↓
7. Validate password
   - Function: core/auth.py → verify_password(submitted_password, stored_hash)
   - Uses Argon2/Bcrypt verification
   - If fails:
     - Record failed attempt
     - Return 401 Unauthorized
   ↓
8. Check auth provider
   - If user.auth_provider == "google":
     - Return 401: "Please use Google Sign-In"
   - Else: Continue
    ↓
9. Successful authentication
   - Clear failed attempts: _login_attempts.pop(email, None)
   - Update user: user.last_login = datetime.utcnow()
   - Save to database: db.save(user)
    ↓
10. Generate JWT token
    - Function: create_access_token({"sub": user.email})
    - Expiry: 24 hours from now
    - Signed with SECRET_KEY
    ↓
11. Return token to frontend
    - Status: 200 OK
    - Body: {
        access_token: "eyJ0eX...",
        token_type: "bearer",
        user: {name, email, created_at}
      }
    ↓
12. Frontend stores token
    - localStorage.setItem('token', data.access_token)
    - localStorage.setItem('user', JSON.stringify(data.user))
    - AuthContext state: setUser(data.user)
    - setLoading(false)
    ↓
13. Frontend navigation
    - navigate('/track')
    - Redirects to protected page
    ↓
✓ User is logged in
```

---

## 7. Admin Dashboard Flow

```
Admin user goes to /admin page
    ↓
1. Frontend loads Admin.jsx
   - No authentication check (SECURITY ISSUE)
   - Sends GET /api/v1/admin/stats
   - Header: x-admin-password: settings.ADMIN_PASSWORD
    ↓
2. Backend receives request
   - Route: admin.py → get_admin_stats()
   - Dependency: verify_admin(header)
   - Checks x_admin_password == settings.ADMIN_PASSWORD
   - If invalid: Return 401 Unauthorized
    ↓
3. Database queries (all run in parallel)
   - Query 1: Total users
     - db.count(User)
     - Returns: integer count
   
   - Query 2: Active users (last 24 hours)
     - twenty_four_hours_ago = utcnow() - timedelta(hours=24)
     - db.count(User, User.last_login >= twenty_four_hours_ago)
   
   - Query 3: Live users (last 2 minutes)
     - two_minutes_ago = utcnow() - timedelta(minutes=2)
     - db.count(ActiveSession, ActiveSession.last_seen_at >= two_minutes_ago)
   
   - Query 4: Total scans
     - db.count(SkinScan)
   
   - Query 5: Recent users (last 10)
     - db.find(User, sort=User.created_at.desc(), limit=10)
   
   - Query 6: Recent scans (last 10)
     - db.find(SkinScan, sort=SkinScan.created_at.desc(), limit=10)
    ↓
4. Backend aggregates results
    - Creates response object: {
        total_users: 150,
        active_users_24h: 45,
        live_users: 8,
        total_scans: 450,
        recent_users: [...],
        recent_scans: [...]
      }
    ↓
5. Backend sends response
    - Status: 200 OK
    - Body: Stats object
    ↓
6. Frontend receives stats
    - Displays summary cards (total users, live users, total scans)
    - Shows recent activity (new users, recent scans)
    ↓
✓ Admin can see system metrics
```

---

## 8. Change Password Flow

```
User on Profile page → clicks "Change Password"
    ↓
1. Form opens (modal or inline)
   - Input: Current Password
   - Input: New Password
   - Input: Confirm New Password
   - Button: Save
    ↓
2. User fills form + clicks Save
   - Frontend validates:
     - newPassword === confirmPassword
     - newPassword meets policy (8+, uppercase, number)
    ↓
3. Frontend calls backend
   - POST /api/v1/users/change-password
   - Headers:
     - Authorization: Bearer <JWT>
     - Content-Type: application/json
   - Body: {current_password, new_password}
    ↓
4. Backend validates request
   - Route: users.py → change_password()
   - Validates JWT → get_current_user()
   - Extracts current user
    ↓
5. Verify current password
   - Function: verify_password(current_password, user.hashed_password)
   - If fails: Return 400 Bad Request: "Incorrect current password"
    ↓
6. Validate new password
   - Function: validate_password(new_password)
   - Checks policy (8+, uppercase, number)
   - If fails: Return 400 Bad Request
    ↓
7. Hash new password
   - Function: get_password_hash(new_password)
   - Result: New Argon2 hash with salt
    ↓
8. Update database
   - user.hashed_password = new_hash
   - db.save(user)
   - MongoDB: UPDATE users SET hashed_password = new_hash WHERE email = user.email
    ↓
9. Return success
   - Status: 200 OK
   - Body: {"message": "Password updated successfully"}
    ↓
10. Frontend displays success
    - Shows confirmation message
    - Clears form
    - Closes modal
    ↓
✓ Password updated
```

---

## State Transformation Points (Key Data Conversions)

| Point | Input Type | Output Type | Transformation |
|---|---|---|---|
| Image Upload | File (browser) | bytes | FileReader.readAsArrayBuffer() |
| Image Compression | File | File | Canvas resize + JPEG encode |
| ML Input | bytes | numpy array | Decode bytes → PIL Image → numpy |
| ML Inference | numpy array (224x224x3) | numpy array (13 floats) | Model.predict() |
| ML Output | numpy array (probabilities) | {disease, confidence, severity} | Argmax + lookup DISEASE_INFO |
| JWT Creation | dict {"sub": email} | string | HS256 encoding |
| JWT Validation | string (token) | dict {sub, exp, iat} | HS256 decoding + exp check |
| Password Hash | string (plaintext) | string (hash) | Argon2 hashing |
| Password Verify | (plaintext, hash) | boolean | Argon2 verification |
| LocalStorage | User object | JSON string | JSON.stringify() |
| Chart Data | [Scan, Scan, Scan] | [{name, confidence, ...}] | Map with formatting |

---

## Error Handling & Fallbacks

```
Image Upload Error
├── Invalid file type → 400 Bad Request
├── File too large → 400 Bad Request
├── Invalid image → 400 Bad Request
└── Network error → "Error connecting to server"

Login Error
├── Email not found → 401 "Invalid credentials"
├── Wrong password → 401 "Invalid credentials"
├── Too many attempts → 429 "Please try again in 5 minutes"
├── Google token invalid → 401 "Invalid Google Access Token"
└── Network error → "Network error or backend unreachable"

Database Error
├── Connection timeout → Logged as warning
├── Query failure → 500 Internal Server Error
└── Save failure → 500 Internal Server Error

ML Model Error
├── Model load failure → Logged on startup
├── Inference failure → 500 "Analysis failed"
├── Model not loaded → Lazy-loads on first request
└── Low confidence → Returns "Unknown" condition

Email Error
├── Invalid email → 400 Bad Request
├── SMTP failure → Logged, email delivery fails silently
└── Resend API failure → Logged error with details
```

---

## Summary: Data Lifecycle Example

```
User Registration Complete Lifecycle:
────────────────────────────────────

1. Browser: User types (name) → React state
2. Browser: User types (email) → React state
3. Browser: User types (password) → React state
4. Browser: User clicks Submit → Form validation
5. Frontend: POST to backend with credentials
6. Network: HTTP request travels to server
7. Backend: Receives JSON
8. Backend: Pydantic validates schema
9. Backend: DB check if email exists
10. Backend: Hash password with Argon2
11. Backend: Create User object
12. Backend: INSERT to MongoDB
13. Backend: Generate JWT with user email
14. Backend: Send JSON response
15. Network: HTTP response travels to browser
16. Frontend: Parse JSON response
17. Frontend: Extract token + user object
18. Frontend: Store in localStorage
19. Frontend: Update React state (AuthContext)
20. Frontend: Redirect to /detect
21. Browser: Render protected page
22. ✓ User can now interact with app

From user action to database record: ~500ms
From database record to UI display: ~500ms
Total journey: ~1 second
```

