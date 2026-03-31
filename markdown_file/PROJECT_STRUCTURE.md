# PROJECT STRUCTURE

Complete breakdown of every file in the SkinAI project with explanations of purpose and importance.

---

## 📁 Root Directory Files

| File/Folder | Purpose |
|---|---|
| `README.md` | Project overview, setup instructions, architecture diagram |
| `docker-compose.yml` | Docker Compose config for local development (frontend + backend) |
| `DOCKER_GUIDE.md` | Documentation for Docker deployment |
| `AWS_DOCKER_DEPLOY_GUIDE.md` | Specific guide for AWS deployment with Docker |
| `digitalocean_deploy.md` | DigitalOcean deployment guide |
| `hosting_guide.md` | General hosting deployment guide |
| `running command.txt` | Quick reference of important commands to run the project |
| `k6-test.js` | Load testing script using k6 (Performance testing) |
| `k6-equivalent.yml` | k6 test config in YAML format |
| `test.py` | Quick Python test script for debugging |
| `fix.py` | Utility script to fix issues (purpose unclear - [UNCLEAR - needs review]) |
| `train_mobilenetv2_colab.py` | Script to train MobileNetV2 model on Google Colab |
| `test-live-users.yml` | Test scenario for live user simulation |
| `test-predict.yml` | Test scenario for prediction endpoint |
| `test-signup.yml` | Test scenario for signup flow |
| `DenseNet121.h5` | Pre-trained DenseNet121 model weights (unused - replaced by MobileNetV2) |
| `DenseNet121.weights.h5` | Additional DenseNet121 weights file (unused) |
| `model_details.md` | Documentation about ML models and performance |
| `query` | File/folder (unclear purpose - [UNCLEAR - needs review]) |

---

## 📁 Backend: `backend/`

### Root Backend Files

| File | Purpose |
|---|---|
| `requirements.txt` | Python dependencies (FastAPI, TensorFlow, Motor, ODMantic, etc.) |
| `Dockerfile` | Docker image definition for backend container |
| `Procfile` | Heroku deployment config |
| `runtime.txt` | Python version specification for deployment |
| `check_env.py` | Script to verify environment variables setup |
| `check_scans.py` | Script to inspect scan records in MongoDB |
| `check_user.py` | Script to inspect user records in MongoDB |
| `test_cloudinary.py` | Unit test for Cloudinary image upload |
| `test_db_connection.py` | Test MongoDB connection |
| `test_db_insecure.py` | Alternative DB test (insecure mode) |
| `test_db_sync.py` | Test synchronous DB operations |
| `test_db.py` | Main DB connection test |
| `test_email.py` | Test email sending via Resend |
| `test_load.py` | Load testing script |
| `test_route.py` | Test API route connectivity |
| `test_signup_retry.py` | Test signup retry logic |
| `test_smtp_direct.py` | Direct SMTP test for email |
| `error_log.txt` | Application error log |
| `scans_log.txt` | Log of scan operations |
| `scans_output.txt` | Output records from scans |

### `backend/app/` - Main Application

#### `main.py` - FastAPI Entry Point
```
Purpose: Define main FastAPI application instance, middleware, error handlers
Contents:
  - Logger setup (structured logging)
  - Rate limiter configuration (SlowAPI)
  - CORS middleware (allows frontend origin)
  - Gzip compression middleware
  - Response time logging middleware
  - Custom exception handlers (RateLimitExceeded, HTTPException)
  - Route inclusion (/api/v1 routers)
  - Startup event (pre-load ML model)
  - Shutdown event (close MongoDB connection)
  - Health check endpoint (/health)
  - Root endpoint (/)
```

---

### `backend/app/core/` - Core Infrastructure

| File | Purpose |
|---|---|
| `config.py` | Environment variables (Pydantic Settings) - DB URL, SECRET_KEY, API keys, etc. |
| `constants.py` | Global constants (DAILY_SCAN_LIMIT, OTP_EXPIRY_MINUTES, PASSWORD_MIN_LENGTH, etc.) |
| `auth.py` | Authentication utilities (password hashing, JWT creation, token verification) |
| `database.py` | MongoDB connection setup (Motor async client, ODMantic engine) |
| `cloudinary_config.py` | Cloudinary SDK initialization |

---

### `backend/app/models/` - Data Models (ODMantic)

| File | Purpose |
|---|---|
| `user.py` | **User** model - Stores user accounts (name, email, hashed_password, auth_provider, is_verified, created_at, last_login) |
| `domain.py` | Domain models - **SkinScan** (stores scan results), **ActiveSession** (tracks online users), **OTPLog** (OTP verification) |

---

### `backend/app/api/` - API Routes

#### `main.py` - Router Registry
```
Purpose: Aggregate all route blueprints into main api_router
Includes:
  - Auth router (prefix: /auth)
  - Users router (prefix: /users)
  - Scan router (prefix: /scan)
  - Doctors router (prefix: /doctors)
  - Admin router (included in main.py as separate endpoint)
```

#### `deps.py` - Dependency Injection
```
Functions:
  - get_current_user(): Extracts JWT from Authorization header, validates token, returns User
  - oauth2_scheme: OAuth2 password bearer token extractor
```

### `backend/app/api/routes/` - Endpoint Handlers

#### `auth.py` - Authentication Endpoints
```
Functions:
  1. generate_otp() → Returns random 6-digit string
  2. validate_password() → Enforces password policy (8+ chars, uppercase, number)
  3. signup(name, email, password) → POST /signup
     - Validates password, checks email uniqueness
     - Hashes password with Argon2
     - Creates User record, issues JWT immediately
     - Returns access_token + user info
  4. login(email, password) → POST /login
     - Implements brute force protection (5 attempts / 5 minutes)
     - Validates credentials, issues JWT
     - Updates last_login timestamp
  5. google_login(token) → POST /google
     - Validates Google access token (calls Google API)
     - Creates/finds user by email
     - Returns JWT + user info
  6. test_email_diag(email) → GET /test-email
     - Diagnostic endpoint to test email delivery
```

#### `users.py` - User Management Endpoints
```
Functions:
  1. validate_password() → Same as auth.py (validates policy)
  2. user_pulse(current_user) → POST /pulse
     - Updates/creates ActiveSession with current timestamp
     - Used for tracking real-time user activity
  3. read_users_me() → GET /me
     - Returns current authenticated user's info
  4. change_password(current_password, new_password, current_user) → POST /change-password
     - Verifies current password, validates new password, updates DB
  5. delete_account(current_user) → DELETE /account
     - Deletes user + all their scans from MongoDB
```

#### `scan.py` - Skin Disease Detection Endpoints
```
Functions:
  1. _upload_to_cloudinary_sync(image_bytes, scan_id) → Background task
     - Uploads image to Cloudinary, updates DB with URL
  2. predict_disease(file) → POST /predict
     - Validates image (type, size, integrity)
     - Calls ml_predict() to run inference
     - Returns disease, confidence, severity, description, recommendations
     - Triggers background Cloudinary upload
  3. get_scan_history(current_user) → GET /history
     - Returns all user's scans sorted by created_at descending
  4. get_upload_limit(current_user) → GET /limit
     - Returns remaining daily scans (DAILY_SCAN_LIMIT = 5)
```

#### `doctors.py` - Doctor Finder Endpoints
```
Classes:
  - Coordinates: latitude, longitude

Functions:
  1. get_nearby_doctors(coords) → POST /nearby
     - Calls find_nearby_dermatologists() service
     - Returns list of doctors with name, address, rating, location
```

#### `admin.py` - Admin Dashboard Endpoints
```
Functions:
  1. verify_admin(x_admin_password) → Header dependency
     - Validates admin password from header
  2. get_admin_stats() → GET /stats
     - Returns total_users, active_users_24h, live_users
     - Returns recent_users (last 10) and recent_scans (last 10)
     - Uses ActiveSession to count live users (last 2 minutes)
```

---

### `backend/app/services/` - Business Logic

#### `ml_model.py` - Machine Learning Inference
```
Global Variables:
  - _model: Cached MobileNetV2 model instance
  - _H5_PATH: Path to model.weights.h5
  - IMAGE_SIZE: 224x224 pixels
  - NUM_CLASSES: 13 disease categories
  - CLASS_NAMES: ['Acne', 'Actinic_Keratosis', ..., 'Unknown', 'Vitiligo', 'Warts']
  - MIN_CONFIDENCE_THRESHOLD: 0.35 (35%)
  - CLASS_MERGE_MAP: Empty (no merging - already merged during training)
  - MERGED_CLASS_INCLUDES: Info about sub-conditions (Fungal_Infection, Benign_Growth)
  - DISEASE_INFO: Dictionary with disease details, severity, do/don't lists

Functions:
  1. _get_model() → Loads MobileNetV2 from HDF5, caches in _model
  2. predict(image_bytes) → Runs inference
     - Decodes image bytes → Resizes to 224x224
     - Normalizes pixel values → Runs model prediction
     - Returns disease, confidence, severity, description, recommendations
     - Returns 'Unknown' if confidence < MIN_CONFIDENCE_THRESHOLD
```

#### `email.py` - Email Service
```
Functions:
  1. send_otp_email(email, otp) → Resend API
     - Sends OTP verification email via Resend
     - HTML template with 10-minute expiry message
  2. send_welcome_email(email, name) → Resend API
     - Sends welcome email to new users
     - Optional (skipped if RESEND_API_KEY not set)
```

#### `maps.py` - Doctor Location Service
```
Functions:
  1. find_nearby_dermatologists(lat, lng, radius=5000) → AWS Location API
     - Calls Amazon Geo Places v2 API
     - Searches for "dermatologist" near coordinates
     - Returns doctor name, address, rating, location
     - Returns mock data if AWS_MAPS_API_KEY not set
```

---

### `backend/tests/` - Unit Tests

| File | Purpose |
|---|---|
| `conftest.py` | Pytest configuration and fixtures |
| `test_auth.py` | Tests for authentication (signup, login, JWT) |
| `test_ml_model.py` | Tests for ML inference |

---

### `backend/tmp/` - Temporary Files
```
Contains: venv_broken_230105 (old broken virtual environment - DELETE)
```

---

## 📁 Frontend: `frontend/`

### Root Frontend Files

| File | Purpose |
|---|---|
| `package.json` | NPM dependencies (React, Vite, Tailwind, Framer Motion, etc.), build scripts |
| `vite.config.js` | Vite build configuration (React plugin, environment variables) |
| `tailwind.config.js` | Tailwind CSS customization |
| `postcss.config.js` | PostCSS plugins (Tailwind processing) |
| `eslint.config.js` | ESLint rules for code quality |
| `index.html` | HTML entry point (loads React app) |
| `nginx.conf` | Nginx config for production web server |
| `Dockerfile` | Docker image for frontend |
| `README.md` | Frontend-specific documentation |
| `update_bgs.py` | Python script to update background images (unclear purpose - [UNCLEAR - needs review]) |

---

### `frontend/src/` - React Application

#### `main.jsx` - Entry Point
```
Purpose: Initialize React app with providers
Includes:
  - ReactDOM.createRoot()
  - HelmetProvider (meta tags)
  - GoogleOAuthProvider (Google Sign-In)
  - BrowserRouter (routing)
  - AuthProvider (authentication context)
```

#### `App.jsx` - Main Router
```
Components:
  - Navbar (top navigation)
  - Routes (all page routes)
  - Footer (bottom navigation)
  - ErrorBoundary (error handling)

Routes:
  Public:
    - / → Landing
    - /login → Login (redirects to /track if authenticated)
    - /signup → Signup (redirects to /track if authenticated)
    - /otp → OTP verification (to be implemented)
    - /success → Success page
    - /privacy → Privacy policy

  Protected (require authentication):
    - /detect → AI skin disease detection
    - /track → Scan history tracker
    - /profile → User profile
    - /doctors → Doctor finder

  Admin:
    - /admin → Admin stats dashboard (no protection [SECURITY - needs review])
```

#### `index.css` & `App.css` - Global Styles
```
Includes:
  - CSS variables (--bg-base, --blue-soft, --white-full, etc.)
  - Tailwind imports
  - Custom animations
  - Day/Night theme variables
```

---

### `frontend/src/context/AuthContext.jsx` - Global Auth State

```
Functions:
  - isTokenExpired(token) → Decodes JWT and checks expiry
  
AuthProvider (context + hook):
  State:
    - user (null or {name, email, created_at})
    - loading (boolean)

  Methods (async):
    - login(email, password) → Calls /auth/login, stores JWT + user
    - signup(name, email, password) → Calls /auth/signup, stores JWT + user
    - googleLogin(token) → Calls /auth/google, stores JWT + user
    - verifyOtp(email, otp) → Calls /auth/verify-otp (NOT IMPLEMENTED)
    - logout() → Clears localStorage, sets user to null

  Effects:
    - On mount: Restore session from localStorage if token exists
    - Every 60 seconds: Send pulse (heartbeat) to /users/pulse if user is logged in

  useAuth() hook: Returns {user, login, signup, logout, googleLogin, verifyOtp, loading}
```

---

### `frontend/src/pages/` - Page Components

| File | Purpose |
|---|---|
| `Landing.jsx` | Public homepage with feature carousel and hero section |
| `Login.jsx` | Email/password login form + Google Sign-In button |
| `Signup.jsx` | Email/password signup form + Google Sign-In, password validation |
| `OTP.jsx` | OTP verification page (to be completed) |
| `Success.jsx` | Success confirmation page after signup |
| `Detect.jsx` | Main feature: Upload skin image → Get AI diagnosis |
| `Tracker.jsx` | View scan history as grid or timeline chart |
| `Profile.jsx` | User info, change password, delete account, scan history |
| `Doctors.jsx` | Find nearby dermatologists by location |
| `Admin.jsx` | Admin dashboard with statistics |
| `Privacy.jsx` | Privacy policy page |

#### `Detect.jsx` - Key Feature Component
```
State:
  - file (selected image file)
  - preview (image preview URL)
  - isAnalyzing (loading state)
  - result (prediction result)
  - error (error message)
  - uploadLimit (remaining daily scans)

Functions:
  - onDrop() → Dropzone callback
  - compressImage() → Reduces image size (800x800, 80% quality)
  - handleAnalyze() → Uploads image → Calls /scan/predict → Returns diagnosis
  - handleDownloadPDF() → Generates PDF report

Flow:
  1. User drops/selects image
  2. Frontend compresses it
  3. Sends FormData to /scan/predict with JWT
  4. Displays severity-color-coded results
  5. User can download PDF report
```

#### `Tracker.jsx` - History Visualization
```
State:
  - history (array of all user's scans)
  - loading (boolean)
  - viewMode ('grid' or 'chart')

Functions:
  - fetchHistory() → GET /scan/history with JWT
  - chartData (computed) → Converts scans to chart format
  - stats (computed) → Total scans, avg confidence, unique conditions

Views:
  - Grid: Card layout showing each scan with disease, confidence, severity
  - Chart: LineChart from Recharts showing confidence trends over time
```

#### `Profile.jsx` - User Account Management
```
Features:
  - Display user info (name, email, member since)
  - Change password form
  - Delete account button
  - Scan history (same as Tracker page)
  - Download PDF reports for each scan

Functions:
  - fetchHistory() → GET /scan/history
  - handleChangePassword() → POST /users/change-password
  - handleLogout() → Calls logout from AuthContext
  - downloadMedicalReport() → Generates PDF for specific scan
```

---

### `frontend/src/components/` - Reusable Components

#### `common/` Directory
| File | Purpose |
|---|---|
| `Navbar.jsx` | Top navigation bar (logo, nav links, auth status, theme toggle) |
| `Footer.jsx` | Bottom footer with links and copyright |
| `ErrorBoundary.jsx` | React error boundary to catch component errors |
| `DermAuraLogo.jsx` | Logo component (SVG or image) |

#### `auth/` Directory
| File | Purpose |
|---|---|
| `AuthLayout.jsx` | Wrapper layout for login/signup pages (form container) |

#### `landing/` Directory
| File | Purpose |
|---|---|
| `Hero.jsx` | Hero section on landing page (headline, CTA button) |
| `FeatureCarousel.jsx` | Carousel showing app features (Framer Motion animations) |

#### `dashboard/` Directory
```
(Specific components for dashboard - to be explored)
```

#### `animations/` Directory
| File | Purpose |
|---|---|
| `ScanningLoader.jsx` | Animated loader shown during ML inference |
| `DNAHelix.jsx` | 3D DNA helix animation (Three.js + React Three Fiber) |

#### Root Component Files
| File | Purpose |
|---|---|
| `DoctorFinder.jsx` | Component to search doctors by location (uses geolocation + /api/v1/doctors/nearby) |
| `DayNightToggle.jsx` | Theme toggle button (light/dark mode) |

---

### `frontend/src/utils/` - Utility Functions

| File | Purpose |
|---|---|
| `pdfGenerator.js` | Generates PDF medical reports using jsPDF + autoTable |
| `cn.js` | Helper function to merge Tailwind CSS classes (clsx + twMerge) |

---

### `frontend/public/` - Static Assets
```
Contains: Fonts, images, icons, animations (static files served as-is)
```

---

## 📁 ML Model Files: `Model_HAM10000/` & `trained_artifacts/`

| File | Purpose |
|---|---|
| `skin_model_best.h5` | Best trained MobileNetV2 model checkpoint |
| `skin_model_best.keras` | Same model in Keras format |
| `model.weights.h5` | Model weights (used by backend) |
| `phase1_log.csv` | Training log (phase 1) |
| `phase2_log.csv` | Training log (phase 2) |
| `classification_report.txt` | Model performance metrics per class |
| `h5tokeras.ipynb` | Jupyter notebook to convert HDF5 to Keras format |
| `MobileNetV2_finetune_log.csv` | Fine-tuning training log |
| `MobileNetV2_head_log.csv` | Head layer training log |

---

## 🚩 Files Flagged as Unused or Unclear

| File | Status | Reason |
|---|---|---|
| `DenseNet121.h5` | ❌ UNUSED | Replaced by MobileNetV2 |
| `DenseNet121.weights.h5` | ❌ UNUSED | Replaced by MobileNetV2 |
| `backend/tmp/venv_broken_230105/` | ❌ UNUSED | Old broken virtual environment, can be deleted |
| `fix.py` | 🟡 UNCLEAR | Purpose not documented |
| `query` | 🟡 UNCLEAR | File/folder purpose unknown |
| `update_bgs.py` | 🟡 UNCLEAR | Python script but purpose not clear |
| `/admin` route | 🔴 SECURITY ISSUE | No authentication check (verify_admin only checks header) |
| `OTP.jsx` page | 🟡 INCOMPLETE | Endpoint exists but integration not finished |

---

## Summary by Role

### Configuration & Setup
- `requirements.txt`, `package.json`, `vite.config.js`, `tailwind.config.js`, `.env`

### Authentication
- `auth.py`, `AuthContext.jsx`, `Login.jsx`, `Signup.jsx`, `deps.py`

### Core Business Logic
- `ml_model.py`, `Detect.jsx`, `scan.py` routes

### User Management
- `User` model, `users.py` routes, `Profile.jsx`

### Data & Database
- `database.py`, Domain models (User, SkinScan, ActiveSession, OTPLog)

### UI/UX
- All components in `frontend/src/components/`
- All pages in `frontend/src/pages/`
- `index.css`, `App.css`

### External Integrations
- `cloudinary_config.py`, `email.py`, `maps.py`

### DevOps & Deployment
- `Dockerfile` (both frontend & backend), `docker-compose.yml`, deployment guides

### Testing
- `backend/tests/`, test scripts under `backend/`

---

## File Count Summary
- **Backend Python files**: ~25 core files
- **Frontend React files**: ~30+ component/page files
- **Configuration files**: ~10
- **Test files**: ~15+
- **ML model files**: ~8
- **Documentation files**: ~10
- **Total project files**: ~100+

