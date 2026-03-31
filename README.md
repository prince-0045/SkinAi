# SkinAi – Curve + Flowchart (Minimal)

This README shows only the requested dataset graph + architecture flowchart, and removes all extra images and verbose sections.

## 📈 Model Graph (from effnetv2s_kaggle_dataset)

![Training and Validation Curves](effnetv2s_kaggle_dataset/training_curves.png)

> Single graph from `effnetv2s_kaggle_dataset`, no other images.

## 🏗️ Architecture Flowchart

```mermaid
graph TB
    subgraph "Frontend (React + Vite)"
        A[Landing Page] --> B[Auth System]
        B --> C[Dashboard]
        C --> D[Image Upload]
        D --> E[Detection Results]
        E --> F[Progress Tracker]
        F --> G[Doctor Finder]
    end

    subgraph "Backend (FastAPI)"
        H[JWT Auth] --> I[User Management]
        I --> J[Scan Processing]
        J --> K[ML Inference]
        K --> L[Report Generation]
        L --> M[Email Service]
    end

    subgraph "ML Pipeline"
        K --> N[EfficientNetV2S Model]
        N --> O[13-Class Classification]
        O --> P[Medical Mapping]
    end

    subgraph "External Services"
        Q[(MongoDB Atlas)]
        R[Cloudinary CDN]
        S[Google OAuth]
        T[Resend Email]
        U[AWS Location]
    end

    Frontend -->|REST API| Backend
    Backend --> Q
    Backend -->|Image Storage| R
    Frontend --> S
    Backend --> T
    Backend --> U
```
- **🩺 AI Skin Disease Detection**: Advanced MobileNetV2 model classifies 13 skin conditions with high accuracy
- **📈 Healing Progress Tracker**: Visual side-by-side comparisons with percentage improvement metrics
- **🔒 Secure Authentication**: Google OAuth + Email/OTP verification with JWT tokens
- **📱 Modern PWA**: Responsive React interface with dark/light mode and smooth animations
- **🏥 Doctor Finder**: Location-based dermatologist search using AWS Geo Services
- **📄 PDF Reports**: Professional medical reports with scan history and recommendations
- **☁️ Cloud Infrastructure**: MongoDB Atlas, Cloudinary CDN, and scalable deployment

## 🚀 Features

### Core Functionality
- **Instant Disease Detection**: Upload skin images for AI-powered analysis
- **13 Disease Classes**: Acne, Actinic Keratosis, Benign Growth, Drug Eruption, Eczema, Fungal Infection, Infestations & Bites, Psoriasis, Rosacea, Skin Cancer, Unknown, Vitiligo, Warts
- **Confidence Scoring**: 0-100% confidence with severity assessment (Mild/Moderate/High/Critical)
- **Medical Insights**: Detailed descriptions, treatment recommendations, do's/don'ts
- **Progress Tracking**: Timeline visualization of healing progress
- **Scan History**: Comprehensive history with trend analysis

### User Experience
- **Responsive Design**: Optimized for desktop, tablet, and mobile
- **Dark/Light Mode**: Automatic theme switching with manual toggle
- **Smooth Animations**: Framer Motion powered transitions
- **PWA Support**: Installable web app with offline capabilities
- **Accessibility**: WCAG compliant with screen reader support

### Security & Privacy
- **End-to-End Encryption**: All data encrypted in transit and at rest
- **GDPR Compliant**: User data control with deletion rights
- **Secure Authentication**: Argon2 password hashing, JWT tokens
- **Rate Limiting**: Protection against abuse with SlowAPI

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Frontend (React + Vite)"
        A[Landing Page] --> B[Auth System]
        B --> C[Dashboard]
        C --> D[Image Upload]
        D --> E[Detection Results]
        E --> F[Progress Tracker]
        F --> G[Doctor Finder]
    end

    subgraph "Backend (FastAPI)"
        H[JWT Auth] --> I[User Management]
        I --> J[Scan Processing]
        J --> K[ML Inference]
        K --> L[Report Generation]
        L --> M[Email Service]
    end

    subgraph "ML Pipeline"
        K --> N[MobileNetV2 Model]
        N --> O[13-Class Classification]
        O --> P[Medical Mapping]
    end

    subgraph "External Services"
        Q[(MongoDB Atlas)]
        R[Cloudinary CDN]
        S[Google OAuth]
        T[Resend Email]
        U[AWS Location]
    end

    Frontend -->|REST API| Backend
    Backend --> Q
    Backend -->|Image Storage| R
    Frontend --> S
    Backend --> T
    Backend --> U
```

## 🛠️ Tech Stack

### Frontend
- **React 19.2.0** - Modern UI framework with hooks
- **Vite** - Lightning-fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Production-ready animations
- **React Router 7.12.0** - Declarative routing
- **Lucide React** - Beautiful icon library
- **Three.js + React Three Fiber** - 3D DNA helix animation
- **Recharts** - Interactive charts for scan history
- **jsPDF + jsPDF-AutoTable** - PDF report generation
- **React Dropzone** - Drag-and-drop file uploads
- **React Helmet Async** - Dynamic meta tags
- **@react-oauth/google** - Google Sign-In integration

### Backend
- **FastAPI** - High-performance async web framework
- **Motor** - Official async MongoDB driver
- **ODMantic** - Async document ORM for MongoDB
- **Pydantic** - Data validation and serialization
- **Passlib + Argon2** - Secure password hashing
- **Python-jose** - JWT token handling
- **Fastapi-mail** - Email delivery via Resend
- **Cloudinary** - Image cloud storage and CDN
- **TensorFlow/Keras** - ML model inference
- **OpenCV + Pillow** - Image preprocessing
- **SlowAPI** - Rate limiting middleware
- **Httpx** - Async HTTP client

### Database & Infrastructure
- **MongoDB Atlas** - Cloud-hosted NoSQL database
- **Cloudinary** - Media asset management
- **Google OAuth 2.0** - Third-party authentication
- **Resend** - Email delivery service
- **AWS Location Services** - Geospatial doctor search

### Development Tools
- **Docker & Docker Compose** - Containerization
- **Gunicorn** - WSGI server for production
- **Nginx** - Reverse proxy and static file serving
- **ESLint + Prettier** - Code quality and formatting

## 🤖 AI Model Details

### Model Architecture
- **Base Model**: MobileNetV2 (EfficientNetV2S variant)
- **Input Size**: 224x224 RGB images
- **Output Classes**: 13 skin disease categories
- **Framework**: TensorFlow/Keras
- **Training Dataset**: HAM10000 + custom augmentation
- **Confidence Threshold**: 35% (0.35)

### Disease Classes
1. **Acne** - Common inflammatory skin condition
2. **Actinic Keratosis** - Precancerous skin lesions
3. **Benign Growth** - Non-cancerous skin tumors
4. **Drug Eruption** - Medication-induced skin reactions
5. **Eczema** - Chronic inflammatory dermatitis
6. **Fungal Infection** - Mycosis affecting skin
7. **Infestations & Bites** - Parasitic or insect-related
8. **Psoriasis** - Autoimmune skin disorder
9. **Rosacea** - Facial skin condition with redness
10. **Skin Cancer** - Malignant skin neoplasms
11. **Unknown** - Unclassified conditions
12. **Vitiligo** - Autoimmune pigment loss
13. **Warts** - Viral skin growths

### Model Performance

#### Training Metrics
- **Final Training Accuracy**: 89.4%
- **Final Validation Accuracy**: 78.8%
- **Training Loss**: 0.84
- **Validation Loss**: 1.09
- **Epochs Trained**: 19
- **Learning Rate**: 3e-5 (with potential decay)

#### Training & Validation Curves (EfficientNetV2S from effnetv2s_kaggle_dataset)

![Training and Validation Curves](effnetv2s_kaggle_dataset/training_curves.png)

> Single graph image included from `effnetv2s_kaggle_dataset` as requested (no extra images).


## 📦 Installation & Setup

### Prerequisites
- **Node.js** 18+ and npm
- **Python** 3.8+
- **MongoDB** (local or Atlas)
- **Git** for cloning

### Quick Local Setup

#### 1. Clone Repository
```bash
git clone https://github.com/prince-0045/SkinAi.git
cd SkinAi
```

#### 2. Backend Setup
```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

#### 3. Environment Configuration
Create `.env` file in `backend/` directory:
```env
# Database
MONGODB_URL=mongodb://localhost:27017/skinai
# Or for MongoDB Atlas: mongodb+srv://username:password@cluster.mongodb.net/skinai

# Security
SECRET_KEY=your_super_secret_key_here_generate_random_64_chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Email (Resend)
MAIL_USERNAME=your_resend_api_key
MAIL_PASSWORD=your_resend_api_key
MAIL_FROM=noreply@yourdomain.com
MAIL_PORT=587
MAIL_SERVER=smtp.resend.com

# Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_google_client_secret

# Cloudinary
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# AWS Location (Optional)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
```

#### 4. Start Backend
```bash
uvicorn app.main:app --reload
```
API available at: http://localhost:8000
Docs at: http://localhost:8000/docs

#### 5. Frontend Setup
```bash
cd frontend
npm install
```

Create `.env` file in `frontend/` directory:
```env
VITE_API_URL=http://localhost:8000
VITE_GOOGLE_CLIENT_ID=your_google_client_id.apps.googleusercontent.com
```

#### 6. Start Frontend
```bash
npm run dev
```
App available at: http://localhost:5173

## 🐳 Docker Deployment

### Local Docker Setup
```bash
# Build and run
docker-compose up --build

# Access points
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Production Docker Deployment
```bash
# Build for production
docker-compose -f docker-compose.prod.yml up --build
```

## 📖 API Reference

### Authentication Endpoints

#### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword",
  "full_name": "John Doe"
}
```

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword"
}
```

#### Google OAuth
```http
POST /api/v1/auth/google
Content-Type: application/json

{
  "token": "google_id_token"
}
```

### Scan Endpoints

#### Upload Image for Analysis
```http
POST /api/v1/scan/upload
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data

file: <image_file>
```

Response:
```json
{
  "disease": "Eczema",
  "confidence": 0.87,
  "severity": "Moderate",
  "description": "Eczema is a chronic skin condition...",
  "recommendation": "Consult a dermatologist for proper treatment",
  "do_list": ["Keep skin moisturized", "Avoid irritants"],
  "dont_list": ["Don't scratch", "Avoid hot showers"],
  "scan_id": "64f1a2b3c4d5e6f7g8h9i0j1"
}
```

#### Get Scan History
```http
GET /api/v1/scan/history
Authorization: Bearer <jwt_token>
```

#### Compare Scans
```http
POST /api/v1/scan/compare
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "scan_id_1": "64f1a2b3...",
  "scan_id_2": "64f1a2b3..."
}
```

### User Management

#### Get Profile
```http
GET /api/v1/users/profile
Authorization: Bearer <jwt_token>
```

#### Update Profile
```http
PUT /api/v1/users/profile
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "full_name": "Updated Name",
  "phone": "+1234567890"
}
```

### Doctor Finder
```http
GET /api/v1/doctors/nearby?lat=40.7128&lng=-74.0060&radius=10
Authorization: Bearer <jwt_token>
```

## 🚀 Production Deployment

### Backend Deployment
```bash
# Use Gunicorn for production
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend Deployment
```bash
# Build production bundle
npm run build

# Serve with Nginx or deploy to Vercel/Netlify
```

### Environment Checklist
- [ ] MongoDB Atlas cluster (M10+ recommended)
- [ ] Cloudinary account with sufficient quota
- [ ] Google OAuth credentials
- [ ] Resend email API key
- [ ] Domain SSL certificate
- [ ] Environment variables configured


## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Load Testing
```bash
# Using locust or similar
locust -f tests/load_test.py
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint/Prettier for JavaScript
- Write tests for new features
- Update documentation
- Ensure cross-browser compatibility



**Built with ❤️ for accessible healthcare**

**Environment Configuration (.env)**
Create a `.env` file in the `backend` directory with the following variables:
```env
MONGODB_URL=mongodb://localhost:27017/skinai  # Or your MongoDB Atlas URL
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Email Configuration (for OTP)
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_app_password
MAIL_FROM=your_email@gmail.com
MAIL_PORT=587
MAIL_SERVER=smtp.gmail.com

# Google Auth
GOOGLE_CLIENT_ID=your_google_client_id
```

Run the backend server:
```bash
uvicorn app.main:app --reload
```
The backend will run at `http://localhost:8000`.

### 3. Frontend Setup

Open a new terminal, navigate to the frontend directory, and install dependencies.

```bash
cd frontend
npm install
```

**Environment Configuration (.env)**
Create a `.env` file in the `frontend` directory:
```env
VITE_API_URL=http://localhost:8000
VITE_GOOGLE_CLIENT_ID=your_google_client_id
```

Run the frontend development server:
```bash
npm run dev
```
The application will usually run at `http://localhost:5173`.

## 📖 API Usage Examples

Here are examples of how to consume the SkinAi backend API from external clients.

### 1. Upload a Image for Analysis
**Endpoint:** `POST /api/v1/scan/upload`
**Auth:** Bearer Token

```bash
curl -X POST "http://localhost:8000/api/v1/scan/upload" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/skin-image.jpg"
```

### 2. Retrieve Scan History (with Cache Headers)
**Endpoint:** `POST /api/v1/scan/history`
**Auth:** Bearer Token

```bash
curl -X GET "http://localhost:8000/api/v1/scan/history" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "accept: application/json"
```

---

## 🚀 Production Deployment Guide

For a highly available production environment, follow these best practices rather than the local dev commands.

### Backend (Gunicorn + Uvicorn Workers)
Do not use raw `uvicorn` in production. Run with Gunicorn to manage multiple worker processes and prevent the ML model from blocking connections.
```bash
# Recommended: 4 workers for standard VMs
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

*Ensure `MONGO_URL` points to a scaled cluster (Atlas Tier M10+) and your Cloudinary quota is sufficient.*

### Frontend (Nginx/Vercel)
The standard `npm run dev` is not for production.

1. Build the production application (creates the PWA artifacts):
   ```bash
   npm run build
   ```
2. Serve the `dist/` directory using **Nginx**, **Vercel**, or **Netlify**. Ensure Single Page App (SPA) routing is configured so all paths fall back to `index.html`.


