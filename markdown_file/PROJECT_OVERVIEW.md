# PROJECT OVERVIEW

## What This Project Does

**DERMAURA** (SkinAI) is an AI-powered web application that helps users detect skin diseases from uploaded photos and track their skin health healing progress over time. Users can:

1. **Upload a skin image** and get instant AI-powered analysis showing:
   - Detected disease/condition
   - Confidence score (0-100%)
   - Severity level (Mild, Moderate, High, Critical)
   - Detailed medical description
   - Professional recommendations
   - Do's and Don't's for treatment
   
2. **View scan history** with timeline charts showing confidence trends and condition patterns over time

3. **Find nearby dermatologists** using location-based search (AWS Location Services)

4. **Generate downloadable PDF medical reports** for each scan

5. **Track healing progress** by comparing images side-by-side over time

6. **Manage account** with secure authentication (email/password or Google Sign-In)

---

## Tech Stack

### Frontend
- **React** (v19.2.0) - UI framework
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations and transitions
- **React Router** (v7.12.0) - Client-side routing
- **Lucide React** - Icon library
- **Three.js** & **React Three Fiber** - 3D graphics (DNA helix animation)
- **Recharts** - Data visualization (charts for scan history)
- **jsPDF & jsPDF-AutoTable** - PDF report generation
- **React Dropzone** - File upload handling
- **React Helmet Async** - Meta tag management
- **Google OAuth** (@react-oauth/google) - Google Sign-In integration

### Backend
- **FastAPI** (Python) - REST API framework
- **Motor** - Async MongoDB driver
- **ODMantic** - Async MongoDB ORM
- **Pydantic** - Data validation
- **Passlib + Argon2** - Password hashing
- **Python-jose** - JWT token creation/verification
- **Fastapi-mail** - Email sending (via Resend)
- **Cloudinary** - Image cloud storage and CDN
- **TensorFlow/Keras** - ML model inference
- **OpenCV** & **Pillow** - Image processing
- **SlowAPI** - Rate limiting
- **Httpx** - Async HTTP requests

### Database
- **MongoDB Atlas** - Cloud NoSQL database
- Collections: `users`, `skin_scans`, `active_sessions`, `otp_logs`

### External Services
- **MongoDB Atlas** - Cloud database hosting
- **Cloudinary CDN** - Image storage and delivery
- **Google OAuth 2.0** - Third-party authentication
- **Resend Email API** - Email delivery
- **AWS Location Services** / **AWS Geo Places API** - Find nearby doctors

### ML Model
- **MobileNetV2** fine-tuned for skin disease classification
- Input size: 224x224 pixels
- Output: 13 disease classes
  - Acne, Actinic Keratosis, Benign Growth, Drug Eruption, Eczema
  - Fungal Infection, Infestations & Bites, Psoriasis, Rosacea
  - Skin Cancer, Unknown, Vitiligo, Warts
- Confidence threshold: 35% (0.35)

---

## Core Problem It Solves

1. **Accessibility**: Provides instant skin disease screening without waiting for dermatologist appointments
2. **Early Detection**: Helps users catch skin conditions early for better treatment outcomes
3. **Education**: Educates users with detailed medical information about their conditions
4. **Progress Tracking**: Enables users to visually monitor healing progress over time
5. **Doctor Locating**: Connects users with nearby dermatologists when consultation is needed
6. **Privacy**: Ensures medical data is encrypted and secure

---

## High-Level Architecture Summary

### Frontend Architecture
```
Landing Page (Public)
├── Hero Section + Feature Showcase
├── Desktop/Mobile Responsive
└── Day/Night Mode Toggle

Authentication Pages (Public → Protected)
├── Login (Email + Password / Google Sign-In)
├── Signup (Email + Password / Google Sign-In)
├── OTP Verification (to be implemented)
└── Success Page

Authenticated Features
├── Detect Page (Upload → Predict → View Results)
│   ├── Image Upload (Drag-drop / Click)
│   ├── Image Compression (800x800, 80% quality)
│   ├── ML Inference (Fast Async)
│   └── Display Results (Severity, Confidence, Recommendations)
│
├── Tracker Page (Historical Analysis)
│   ├── Grid View (Card layout of all scans)
│   ├── Chart View (Timeline confidence trend)
│   └── Summary Statistics
│
├── Profile Page
│   ├── User Info Display
│   ├── Change Password
│   ├── Delete Account
│   ├── Scan History
│   └── Download PDF Reports
│
├── Doctors Page (Geolocation)
│   ├── Map (if integrated)
│   └── Doctor List (AWS Geo Places API)
│
└── Admin Page (Private)
    └── Stats Dashboard

Shared Components
├── Navbar (Brand + Auth Status + Theme Toggle)
├── Footer
├── Error Boundary
├── Auth Layout (Auth pages wrapper)
└── Animations (Scanning Loader, DNA Helix, etc.)
```

### Backend Architecture
```
FastAPI App
├── Middleware
│   ├── CORS (Allow frontend origin)
│   ├── Gzip Compression
│   ├── Rate Limiting (1000/minute global)
│   └── Response Time Logging
│
├── Routes
│   ├── /api/v1/auth/ (Authentication)
│   │   ├── POST /signup → Create user
│   │   ├── POST /login → Generate JWT
│   │   ├── POST /google → Google Sign-In
│   │   └── GET /test-email → Diagnostic endpoint
│   │
│   ├── /api/v1/users/ (User Management)
│   │   ├── POST /pulse → Heartbeat (track active users)
│   │   ├── GET /me → Current user info
│   │   ├── POST /change-password → Update password
│   │   └── DELETE /account → Delete user + scans
│   │
│   ├── /api/v1/scan/ (Skin Disease Detection)
│   │   ├── POST /predict → Upload image → Run ML → Return diagnosis
│   │   ├── GET /history → Get all user's past scans
│   │   └── GET /limit → Check daily upload limit
│   │
│   ├── /api/v1/doctors/ (Doctor Finding)
│   │   └── POST /nearby → Search dermatologists by coordinates
│   │
│   └── /api/v1/admin/ (Admin Dashboard)
│       └── GET /stats → Get system statistics
│
├── Core Services
│   ├── auth.py
│   │   ├── verify_password() → Compare password vs hash
│   │   ├── get_password_hash() → Argon2 hash
│   │   └── create_access_token() → Generate JWT (24h expiry)
│   │
│   ├── database.py
│   │   ├── Database.setup_db() → MongoDB connection pooling
│   │   └── get_db() → Dependency injection for DB engine
│   │
│   ├── ml_model.py
│   │   ├── _get_model() → Load MobileNetV2 weights
│   │   ├── predict() → Run inference on image bytes
│   │   └── DISEASE_INFO dict → Medical details per condition
│   │
│   ├── email.py
│   │   ├── send_otp_email() → OTP delivery (Resend API)
│   │   └── send_welcome_email() → Onboarding email
│   │
│   ├── maps.py
│   │   └── find_nearby_dermatologists() → AWS Location API
│   │
│   └── cloudinary_config.py
│       └── Cloudinary client setup
│
├── Data Models (Pydantic + ODMantic)
│   ├── User
│   │   ├── name, email (unique), hashed_password
│   │   ├── auth_provider (email or google)
│   │   ├── is_verified, created_at, last_login
│   │   └── Collection: "users"
│   │
│   ├── SkinScan
│   │   ├── user_id, image_url, disease_detected
│   │   ├── confidence_score, severity_level
│   │   ├── description, recommendation
│   │   ├── do_list, dont_list, created_at
│   │   └── Collection: "skin_scans"
│   │
│   ├── ActiveSession
│   │   ├── user_id, last_seen_at
│   │   └── Collection: "active_sessions"
│   │
│   └── OTPLog
│       ├── email, otp, expires_at, verified
│       └── Collection: "otp_logs"
│
└── Startup/Shutdown
    ├── startup_event() → Pre-load ML model
    └── shutdown_event() → Close MongoDB connection
```

### Data Flow

1. **User Registration/Login**
   - Frontend sends credentials → Backend validates → MongoDB stores hash
   - Backend returns JWT token → Frontend stores in localStorage
   - Frontend sends JWT with all authenticated requests

2. **Image Upload & Analysis**
   - User selects/drops image on Detect page
   - Frontend compresses image (800x800, 80% quality)
   - Frontend sends FormData with JWT to `/api/v1/scan/predict`
   - Backend validates file type/size
   - Backend loads ML model (if not cached) → Runs inference
   - Backend returns prediction (disease, confidence, severity, description, recommendations)
   - Frontend displays results with severity color coding
   - Backend later uploads image async to Cloudinary (background task)

3. **View Scan History**
   - Frontend requests `/api/v1/scan/history` with JWT
   - Backend queries MongoDB for all user's scans
   - Frontend renders cards (grid) or chart (timeline)
   - Charts calculate confidence trend over time

4. **Find Doctors**
   - Frontend gets user's GPS coordinates via browser Geolocation API
   - Frontend sends coords to `/api/v1/doctors/nearby`
   - Backend calls AWS Location Services
   - Backend returns list of nearby dermatologists
   - Frontend displays in list/map

---

## Authentication & Security

- **JWT Tokens**: 24-hour expiration, renewed on login
- **Password Hashing**: Argon2 (secure, slow to compute)
- **Brute Force Protection**: 5 failed attempts = 5 minute lockout per email
- **Rate Limiting**: 1000 requests per minute per IP
- **CORS**: Whitelist specific origins (localhost:5173, dermaura.tech)
- **Image Validation**: File type check, size check (max 10MB), image integrity verify
- **Google OAuth**: Secure token validation via Google API

---

## Deployment

- **Frontend**: Vite build → Deployed to Vercel/Netlify
- **Backend**: Uvicorn server → Deployed to Heroku/Railway/AWS
- **Database**: MongoDB Atlas (cloud-managed)
- **Images**: Cloudinary CDN
- **Containerization**: Docker available (Dockerfile in both frontend & backend)

---

## Key Features by Priority

### ✅ Implemented
1. AI skin disease detection (13-class MobileNetV2)
2. User authentication (email/password + Google Sign-In)
3. Scan history with timeline charts
4. User dashboard (profile, change password, delete account)
5. Doctor finder (AWS Location Services)
6. PDF report generation
7. Admin stats dashboard
8. Rate limiting & security

### 🟡 Partial/To-Be-Enhanced
- OTP email verification (endpoint exists, frontend integration pending)
- Image compression (implemented but can be optimized)
- Mobile-responsive design (mostly complete, further refinement needed)

### ❌ Not Yet Implemented
- Real-time notifications
- Side-by-side image comparison (healing progress tracker)
- Appointment booking integration
- Advanced filtering/sorting in history

