# CONNECTIONS_MAP

Complete dependency map showing which functions call which functions, which files import which files, and critical "hub" files that everything depends on.

---

## Backend Dependency Graph

### Core Hub Files (Everything Depends On These)

#### `backend/app/core/config.py` - Settings Module
```
Imported by: EVERY FILE THAT NEEDS ENVIRONMENT VARIABLES

Dependents:
├── main.py (settings.FRONTEND_URL, rate limiting config)
├── database.py (settings.MONGO_URL)
├── auth.py (settings.SECRET_KEY, settings.ALGORITHM)
├── email.py (settings.MAIL_*, settings.RESEND_API_KEY)
├── maps.py (settings.AWS_MAPS_API_KEY, settings.AWS_REGION)
├── cloudinary_config.py (settings.CLOUDINARY_*)
├── routes/auth.py (settings.ADMIN_PASSWORD)
├── routes/scan.py (settings.FRONTEND_URL)
└── services/ml_model.py (implicit via constants)

Health Impact: CRITICAL - If config fails, entire app fails
```

#### `backend/app/core/constants.py` - Application Constants
```
Imported by:
├── routes/auth.py (OTP_EXPIRY_MINUTES, LOGIN_MAX_ATTEMPTS, PASSWORD_MIN_LENGTH)
├── routes/scan.py (DAILY_SCAN_LIMIT, MAX_UPLOAD_SIZE_BYTES, ALLOWED_IMAGE_TYPES)
├── services/ml_model.py (MIN_CONFIDENCE_THRESHOLD)
└── routes/users.py (PASSWORD_MIN_LENGTH)

Health Impact: MEDIUM - Configuration errors affect validation
```

#### `backend/app/core/database.py` - Database Connection
```
Imported by: EVERY ROUTE THAT NEEDS DB ACCESS

Dependents:
├── routes/auth.py (get_db dependency)
├── routes/users.py (get_db dependency)
├── routes/scan.py (get_db dependency)
├── routes/doctors.py (db global variable)
├── routes/admin.py (db.engine global)
└── api/deps.py (get_db dependency)

Functions called:
├── setup_db() - Called by get_db()
└── get_db() - Used in every route via Depends()

Health Impact: CRITICAL - No DB = No app functionality
```

### Route Layer

#### `backend/app/api/main.py` - Router Registry
```
Contains: api_router = APIRouter()

Imports:
├── from app.api.routes import auth
├── from app.api.routes import users
├── from app.api.routes import scan
└── from app.api.routes import doctors

Exports:
└── api_router (included in main.py as app.include_router())

Dependencies:
└── These are routers, not functions - they're middleware-like

Health Impact: HIGH - If router registration fails, no endpoints available
```

### Authentication Flow

```
frontend/context/AuthContext.jsx
├── login() calls:
│   └── fetch(/api/v1/auth/login)
│       └── backend/routes/auth.py → login()
│           ├── verify_password() [core/auth.py]
│           ├── create_access_token() [core/auth.py]
│           └── db.find_one(User) [database.py]
│
├── signup() calls:
│   └── fetch(/api/v1/auth/signup)
│       └── backend/routes/auth.py → signup()
│           ├── validate_password()
│           ├── get_password_hash() [core/auth.py]
│           ├── create_access_token() [core/auth.py]
│           ├── db.save(User) [database.py]
│           └── send_welcome_email() [services/email.py]
│
└── googleLogin() calls:
    └── fetch(/api/v1/auth/google)
        └── backend/routes/auth.py → google_login()
            ├── httpx.get(google.com/userinfo) [external API]
            ├── create_access_token() [core/auth.py]
            └── db.save(User) [database.py]
```

### Scan/ML Flow

```
frontend/pages/Detect.jsx
├── compressImage() - Local function
└── handleAnalyze() calls:
    └── fetch(/api/v1/scan/predict)
        └── backend/routes/scan.py → predict_disease()
            ├── validate file
            ├── ml_predict() [services/ml_model.py]
            │   ├── _get_model() - Loads cached model
            │   └── model.predict() - TensorFlow inference
            ├── _upload_to_cloudinary_sync() - Background task
            │   └── cloudinary.uploader.upload() [external]
            └── db.save(ScanScan) [database.py]

Get History:
└── fetch(/api/v1/scan/history)
    └── backend/routes/scan.py → get_scan_history()
        └── db.find(ScanScan) [database.py]

Get Limit:
└── fetch(/api/v1/scan/limit)
    └── backend/routes/scan.py → get_upload_limit()
        └── db.count(ScanScan) [database.py]
```

### User Management Flow

```
frontend/pages/Profile.jsx
├── changePassword calls:
│   └── fetch(/api/v1/users/change-password)
│       └── backend/routes/users.py → change_password()
│           ├── verify_password() [core/auth.py]
│           ├── validate_password()
│           ├── get_password_hash() [core/auth.py]
│           └── db.save(User) [database.py]
│
├── deleteAccount calls:
│   └── fetch(/api/v1/users/delete-account)
│       └── backend/routes/users.py → delete_account()
│           ├── db.find(SkinScan) [database.py]
│           ├── db.delete(SkinScan) [database.py]
│           └── db.delete(User) [database.py]
│
├── getUserInfo calls:
│   └── fetch(/api/v1/users/me)
│       └── backend/routes/users.py → read_users_me()
│           └── get_current_user() [api/deps.py]
│               └── jwt.decode() [core/auth.py]
│
└── pulse calls (automatic every 60s):
    └── fetch(/api/v1/users/pulse)
        └── backend/routes/users.py → user_pulse()
            ├── db.find_one(ActiveSession) [database.py]
            └── db.save(ActiveSession) [database.py]
```

### Doctor Finding Flow

```
frontend/pages/Doctors.jsx → DoctorFinder.jsx
└── handleSearch calls:
    └── fetch(/api/v1/doctors/nearby)
        └── backend/routes/doctors.py → get_nearby_doctors()
            └── find_nearby_dermatologists() [services/maps.py]
                ├── returns mock data if no API key
                └── httpx.post(AWS Location API) [external]
```

### Admin Dashboard Flow

```
frontend/pages/Admin.jsx
└── fetchStats calls:
    └── fetch(/api/v1/admin/stats)
        └── backend/routes/admin.py → get_admin_stats()
            ├── verify_admin() - Validates password header
            ├── db.count(User) [database.py]
            ├── db.count(User, filter) [database.py]
            ├── db.count(ActiveSession, filter) [database.py]
            ├── db.count(SkinScan) [database.py]
            ├── db.find(User, limit=10) [database.py]
            └── db.find(SkinScan, limit=10) [database.py]
```

---

## Frontend Dependency Graph

### Core Hub Files

#### `frontend/src/context/AuthContext.jsx` - Global State
```
Imported by: ALMOST EVERY PAGE/COMPONENT

Dependents:
├── App.jsx (uses useAuth for ProtectedRoute)
├── pages/Login.jsx (useAuth for login method)
├── pages/Signup.jsx (useAuth for signup method)
├── pages/Detect.jsx (useAuth for user info)
├── pages/Tracker.jsx (useAuth for user state)
├── pages/Profile.jsx (useAuth for user, logout)
├── pages/Doctors.jsx (useAuth dependency)
├── components/common/Navbar.jsx (useAuth for user display)
├── services throughout app (useAuth)

Functions exported:
├── useAuth() hook
├── AuthProvider component
└── All auth functions (login, signup, logout, etc.)

Health Impact: CRITICAL - Auth failure = no protected pages accessible
```

#### `frontend/src/utils/pdfGenerator.js` - PDF Generation
```
Imported by:
├── pages/Detect.jsx (downloadMedicalReport)
└── pages/Profile.jsx (downloadMedicalReport)

Exports:
└── downloadMedicalReport(result, user)

Health Impact: MEDIUM - PDF feature secondary to core app
```

#### `frontend/src/utils/cn.js` - CSS Utilities
```
Imported by: SOME COMPONENTS FOR CLASSNAME MERGING

Used in:
├── CSS class merging throughout
├── Tailwind + custom class conflicts

Exports:
└── cn(...classes) function

Health Impact: LOW - Optional utility function
```

### Page Layer Dependencies

#### `Detect.jsx` → Core Detection Page
```
Imports:
├── AuthContext (useAuth)
├── pdfGenerator (downloadMedicalReport)
├── framer-motion (motion animations)
├── lucide-react (icons)
├── react-dropzone (file upload)
└── Animation component (ScanningLoader)

Functions:
├── compressImage() - Local helper
├── handleAnalyze() - Calls /api/v1/scan/predict
├── handleDownloadPDF() - Calls downloadMedicalReport()

Depends on backend:
└── /api/v1/scan/predict endpoint

Health Impact: CRITICAL - Core feature
```

#### `Tracker.jsx` → Scan History
```
Imports:
├── AuthContext (useAuth)
├── Recharts (LineChart, XAxis, etc.)
├── lucide-react (icons)
└── framer-motion (animations)

Functions:
├── fetchHistory() - GET /api/v1/scan/history
├── chartData computed - Formats data for chart
└── stats computed - Calculates summary

Depends on backend:
├── /api/v1/scan/history endpoint
└── /api/v1/scan/limit endpoint

Health Impact: HIGH - Key feature for tracking
```

#### `Profile.jsx` → User Settings
```
Imports:
├── AuthContext (logout, useAuth)
├── pdfGenerator (downloadMedicalReport)
├── lucide-react (icons)
└── react-router (useNavigate)

Functions:
├── fetchHistory() - GET /api/v1/scan/history
├── handleChangePassword() - POST /api/v1/users/change-password
├── handleLogout() - Calls AuthContext.logout()

Depends on backend:
├── /api/v1/scan/history
├── /api/v1/users/change-password
└── /api/v1/users/delete-account

Health Impact: HIGH - Account management
```

#### `Login.jsx` → Authentication
```
Imports:
├── AuthContext (login, googleLogin)
├── @react-oauth/google (useGoogleLogin)
├── lucide-react (icons)
└── react-router (useNavigate, Link)

Functions:
├── handleLogin() - AuthContext.login()
├── handleGoogleLogin() - AuthContext.googleLogin()

Depends on backend:
├── /api/v1/auth/login
└── /api/v1/auth/google

Health Impact: CRITICAL - Entry point
```

#### `Signup.jsx` → User Registration
```
Imports:
├── AuthContext (signup, googleLogin)
├── @react-oauth/google (useGoogleLogin)
├── lucide-react (icons)
└── react-router (useNavigate, Link)

Functions:
├── handleSignup() - AuthContext.signup()
├── handleGoogleSignup() - AuthContext.googleLogin()
├── Local password validation

Depends on backend:
├── /api/v1/auth/signup
└── /api/v1/auth/google

Health Impact: CRITICAL - User onboarding
```

#### `Doctors.jsx` → Doctor Finder
```
Imports:
├── DoctorFinder component
├── lucide-react (icons)
└── react-helmet (meta tags)

Uses component:
└── DoctorFinder.jsx

Depends on backend:
└── /api/v1/doctors/nearby

Health Impact: MEDIUM - Secondary feature
```

### Component Dependencies

#### Navigation
```
App.jsx → Navbar.jsx
├── Uses AuthContext (useAuth)
├── Displays user info if logged in
├── Theme toggle (DayNightToggle.jsx)
└── Links to all routes

App.jsx → Footer.jsx
└── Static footer with links

App.jsx → ErrorBoundary.jsx
└── Catches component errors
```

#### Authentication Components
```
AuthLayout.jsx
├── Wrapper for Login.jsx and Signup.jsx
├── Shared styling and layout
├── Input field components (icons from lucide-react)
```

#### Animations
```
Components using animations:
├── Landing.jsx → DNAHelix.jsx (3D animation)
├── Landing.jsx → FeatureCarousel.jsx (sliding cards)
├── Detect.jsx → ScanningLoader.jsx (loader animation)
└── All pages → Framer Motion (transitions)
```

---

## Critical Dependency Chains

### Most Critical Path (If Breaks, App Breaks)

```
1. AuthContext
   └── Manages user session
   └── Required by: App.jsx ProtectedRoute

2. Database Connection (database.py)
   └── Required by: All routes
   └── Required by: Core business logic

3. Config/Constants
   └── Required by: Database, Auth, ML, Email
   └── No config = No variables, no secrets

4. Auth Utilities (core/auth.py)
   └── Required by: All auth routes
   └── No auth = No security

5. ML Model Service (services/ml_model.py)
   └── Required by: /scan/predict endpoint
   └── No model = No predictions

Failure order of impact:
1st: AuthContext/Database → App completely broken
2nd: Config → Crashes on startup
3rd: Auth utilities → Can't login
4th: ML Model → A specific feature (scanning) broken
```

### Integration Points

```
Frontend ←→ Backend
├── HTTP REST API
├── JSON request/response bodies
├── JWT authentication (bearer tokens)
└── CORS headers for cross-origin requests

Backend ←→ Database (MongoDB)
├── Motor async driver
├── ODMantic ORM models
├── Connection pooling (50 max)
└── Collections: users, skin_scans, active_sessions, otp_logs

Backend ←→ External APIs
├── Google OAuth API (verify tokens)
├── AWS Location Services (find doctors)
├── Cloudinary API (upload images)
├── Resend Email API (send emails)
└── TensorFlow/Keras (ML inference)

Frontend ←→ External APIs
├── Google OAuth (sign-in library)
├── Browser Geolocation API (get location)
└── Local Storage (persist session)
```

---

## File Dependency Matrix

### Backend Files

| File | Depends On | Used By |
|---|---|---|
| main.py | config, database, auth, email, ml_model, routes | N/A (entry point) |
| core/config.py | None (loads env) | admin/app files |
| core/auth.py | jose, passlib | routes/auth, routes/users |
| core/database.py | motor, odmantic | all routes |
| core/constants.py | None | routes/auth, routes/scan, services/ml_model |
| api/main.py | api/routes/* | main.py |
| api/deps.py | auth, database | all routes |
| routes/auth.py | models/user, core/*, services/email | api/main.py |
| routes/users.py | models/user, routes/scan, core/* | api/main.py |
| routes/scan.py | models/domain, services/ml_model, cloudinary | api/main.py |
| routes/doctors.py | services/maps | api/main.py |
| routes/admin.py | models/*, database | main.py directly |
| services/ml_model.py | constants | routes/scan |
| services/email.py | config | routes/auth |
| services/maps.py | config, httpx | routes/doctors |
| models/user.py | odmantic | routes/auth, routes/users, auth |
| models/domain.py | odmantic | routes/scan, routes/admin |

### Frontend Files

| File | Depends On | Used By |
|---|---|---|
| main.jsx | AuthContext, App | N/A (entry point) |
| App.jsx | All pages, AuthContext, Navbar, Footer | N/A (router) |
| context/AuthContext.jsx | fetch, localStorage | All authenticated pages |
| pages/Login.jsx | AuthContext, @react-oauth | App router |
| pages/Signup.jsx | AuthContext, @react-oauth | App router |
| pages/Detect.jsx | AuthContext, pdfGenerator | App router |
| pages/Tracker.jsx | AuthContext | App router |
| pages/Profile.jsx | AuthContext, pdfGenerator | App router |
| pages/Doctors.jsx | DoctorFinder | App router |
| pages/Admin.jsx | None (insecure) | App router |
| DoctorFinder.jsx | fetch, geolocation API | Doctors.jsx |
| pdfGenerator.js | jsPDF, jspdf-autotable | Detect.jsx, Profile.jsx |
| cn.js | clsx, tailwind-merge | Optional utility |
| Navbar.jsx | AuthContext, DayNightToggle | App.jsx |

---

## Hub Files Summary

### Backend Hubs
```
Tier 1 (Absolute Foundation):
├── main.py - Express app entry
├── core/config.py - Environment config
└── core/database.py - MongoDB connection

Tier 2 (Security/Core Logic):
├── core/auth.py - JWT, password hashing
├── services/ml_model.py - AI inference
└── api/deps.py - Auth dependency injection

Tier 3 (Routes/Features):
├── routes/auth.py - User authentication
├── routes/scan.py - Main feature (scanning)
├── routes/users.py - Account management
└── routes/doctors.py - Doctor finding

Tier 4 (Support Services):
├── services/email.py - Email delivery
├── services/maps.py - Location lookup
└── cloudinary_config.py - Image storage
```

### Frontend Hubs
```
Tier 1 (Absolute Foundation):
├── main.jsx - React entry point
└── context/AuthContext.jsx - Global state

Tier 2 (Core Pages):
├── App.jsx - Router and layout
├── pages/Login.jsx - Entry point
└── pages/Signup.jsx - Onboarding

Tier 3 (Main Features):
├── pages/Detect.jsx - Scanning (core)
├── pages/Tracker.jsx - History
├── pages/Profile.jsx - Account
└── pages/Doctors.jsx - Finding doctors

Tier 4 (UI Components):
├── Navbar.jsx - Navigation
├── DoctorFinder.jsx - Search
└── Animations and common components

Tier 5 (Utilities):
├── pdfGenerator.js - PDF export
└── cn.js - CSS helpers
```

---

## Potential Breaking Points

| Component | Impact | If Broken |
|---|---|---|
| AuthContext (frontend) | CRITICAL | Can't login, all protected routes fail |
| main.py (backend) | CRITICAL | App doesn't start |
| database.py | CRITICAL | No data persistence |
| config.py | CRITICAL | Missing secrets, can't connect to services |
| core/auth.py | HIGH | Can't generate/validate tokens |
| services/ml_model.py | HIGH | Scanning feature fails |
| routes/auth.py | HIGH | Can't login/signup |
| routes/scan.py | HIGH | Can't scan images |
| App.jsx (frontend) | HIGH | Router broken, can't navigate |
| pdfGenerator.js | MEDIUM | Can't download reports (non-critical) |
| services/email.py | MEDIUM | Can't send emails, workflows still work |
| services/maps.py | LOW | Doctor finder broken, app still works |

---

## Module Interdependencies 

```
Highest Coupling (Most imports/exports):
1. core/database.py ← Used by everything
2. core/auth.py ← Used by all auth/user routes
3. core/config.py ← Used by all services
4. context/AuthContext.jsx ← Used by all pages

Moderate Coupling:
5. services/ml_model.py ← Used by scan route
6. models/user.py ← Used by auth routes
7. routes/auth.py ← Complex business logic
8. App.jsx ← Depends on many pages

Low Coupling (Can be modified independently):
- Utility functions (pdfGenerator, cn)
- Decorative components (DayNightToggle, animations)
- Support services (email, maps)
- Landing page
```

