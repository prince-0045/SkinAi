# 🚀 SkinAI - Full Website Deployment Guide (Docker + AWS)

> **Stack**: React/Vite Frontend + FastAPI Backend + model.h5 + MongoDB Atlas + Cloudinary  
> **AWS Services Used**: ECR, ECS (Fargate), ALB, Route 53, ACM (SSL), S3 + CloudFront

---

## 📐 Architecture Overview

```
User Browser
     │
     ▼
CloudFront (CDN) ──────► S3 Bucket (React Frontend)
     │
     ▼
Application Load Balancer (ALB)
     │
     ▼
ECS Fargate Cluster
     ├── Backend Container (FastAPI + model.h5)
     │        │
     │        ├──► MongoDB Atlas (Database)
     │        └──► Cloudinary (Image Storage)
     │
ECR (Docker Image Registry)
```

---

## 🧰 Prerequisites — Install These First

| Tool | Purpose | Download |
|------|---------|----------|
| AWS CLI | Control AWS from terminal | [aws.amazon.com/cli](https://aws.amazon.com/cli/) |
| Docker Desktop | Build & test containers | [docker.com](https://docker.com) |
| AWS Account | Free tier available | [aws.amazon.com](https://aws.amazon.com) |

### Verify installations
```powershell
aws --version        # should show: aws-cli/2.x.x
docker --version     # should show: Docker version 24.x.x
```

### Configure AWS CLI
```powershell
aws configure
# Enter:
# AWS Access Key ID: (from AWS Console → IAM → Your User → Security Credentials)
# AWS Secret Access Key: (same place)
# Default region: ap-south-1    ← Mumbai (close to India, low latency)
# Default output format: json
```

---

## 📦 PART 1 — Dockerize the Backend (FastAPI)

### Step 1.1 — Create `backend/Dockerfile`

Create this file at `d:\SkinAi\backend\Dockerfile`:

```dockerfile
# Use Python 3.10 slim base
FROM python:3.10-slim

WORKDIR /app

# Install system libraries needed by OpenCV and TensorFlow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code + model file
COPY . .

# Expose port 8000
EXPOSE 8000

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 1.2 — Create `backend/.dockerignore`

Create this file at `d:\SkinAi\backend\.dockerignore`:

```
venv/
__pycache__/
*.pyc
*.pyo
.env
*.log
*.txt
!requirements.txt
test_*.py
check_*.py
```

> ⚠️ **Important**: `.env` is excluded from Docker. You will inject env vars via AWS Secrets Manager later.

### Step 1.3 — Test the Backend Docker Image Locally

```powershell
cd d:\SkinAi\backend

# Build the image
docker build -t skinai-backend:latest .

# Run it locally (pass your .env variables)
docker run -p 8000:8000 `
  -e MONGO_URL="your_mongo_url" `
  -e SECRET_KEY="your_secret" `
  -e FRONTEND_URL="http://localhost:5173" `
  skinai-backend:latest

# Test it
# Open browser: http://localhost:8000
# Should see: {"message": "Welcome to SkinCare AI API"}
```

---

## 🎨 PART 2 — Dockerize the Frontend (React/Vite)

### Step 2.1 — Create `frontend/Dockerfile`

Create this file at `d:\SkinAi\frontend\Dockerfile`:

```dockerfile
# Stage 1: Build the React app
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Build the production bundle
# The backend URL is injected at build time
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Copy built files to nginx
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx config to handle React Router
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### Step 2.2 — Create `frontend/nginx.conf`

Create this file at `d:\SkinAi\frontend\nginx.conf`:

```nginx
server {
    listen 80;
    server_name _;

    root /usr/share/nginx/html;
    index index.html;

    # Handle React Router (all routes serve index.html)
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### Step 2.3 — Create `frontend/.dockerignore`

Create this file at `d:\SkinAi\frontend\.dockerignore`:

```
node_modules/
dist/
.env
.env.local
```

### Step 2.4 — Test the Frontend Docker Image Locally

```powershell
cd d:\SkinAi\frontend

# Build the image (point to your local backend for testing)
docker build `
  --build-arg VITE_API_URL=http://localhost:8000/api/v1 `
  -t skinai-frontend:latest .

# Run it
docker run -p 3000:80 skinai-frontend:latest

# Open browser: http://localhost:3000
```

### Step 2.5 — Test Both Together with Docker Compose (Optional but Recommended)

Create `d:\SkinAi\docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - MONGO_URL=${MONGO_URL}
      - SECRET_KEY=${SECRET_KEY}
      - FRONTEND_URL=http://localhost:3000
      - CLOUDINARY_CLOUD_NAME=${CLOUDINARY_CLOUD_NAME}
      - CLOUDINARY_API_KEY=${CLOUDINARY_API_KEY}
      - CLOUDINARY_API_SECRET=${CLOUDINARY_API_SECRET}

  frontend:
    build:
      context: ./frontend
      args:
        VITE_API_URL: http://localhost:8000/api/v1
    ports:
      - "3000:80"
    depends_on:
      - backend
```

```powershell
# Run both services
cd d:\SkinAi
docker-compose up --build

# Visit http://localhost:3000  ← full app
```

---

## ☁️ PART 3 — Push Images to AWS ECR (Container Registry)

### Step 3.1 — Create ECR Repositories

```powershell
# Create backend repository
aws ecr create-repository --repository-name skinai-backend --region ap-south-1

# Create frontend repository
aws ecr create-repository --repository-name skinai-frontend --region ap-south-1
```

> Copy the `repositoryUri` from the output. Looks like:
> `123456789.dkr.ecr.ap-south-1.amazonaws.com/skinai-backend`

### Step 3.2 — Login to ECR

```powershell
# Get your AWS account ID
aws sts get-caller-identity --query Account --output text

# Login Docker to ECR (replace YOUR_ACCOUNT_ID)
aws ecr get-login-password --region ap-south-1 | `
  docker login --username AWS --password-stdin `
  YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com
```

### Step 3.3 — Tag and Push Backend Image

```powershell
$ACCOUNT_ID = aws sts get-caller-identity --query Account --output text
$ECR_BASE = "$ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com"

# Tag the image
docker tag skinai-backend:latest $ECR_BASE/skinai-backend:latest

# Push to ECR
docker push $ECR_BASE/skinai-backend:latest
```

### Step 3.4 — Tag and Push Frontend Image

```powershell
# Build with your real backend URL (we'll update after backend deploys)
docker build `
  --build-arg VITE_API_URL=https://api.yourdomain.com/api/v1 `
  -t skinai-frontend:latest `
  d:\SkinAi\frontend

# Tag and push
docker tag skinai-frontend:latest $ECR_BASE/skinai-frontend:latest
docker push $ECR_BASE/skinai-frontend:latest
```

---

## 🖥️ PART 4 — Deploy Backend on ECS Fargate

> ECS Fargate = AWS-managed containers (no servers to manage, pay per second)

### Step 4.1 — Create an ECS Cluster

1. Open **AWS Console** → Search **ECS** → Click **Clusters** → **Create Cluster**
2. **Cluster name**: `skinai-cluster`
3. **Infrastructure**: ✅ AWS Fargate (serverless)
4. Click **Create**

### Step 4.2 — Store Secrets in AWS Parameter Store

Never put secrets in environment variables directly. Use AWS SSM:

```powershell
# Store each secret (replace values with your real ones)
aws ssm put-parameter --name "/skinai/MONGO_URL" `
  --value "mongodb+srv://user:pass@cluster.mongodb.net/skinai" `
  --type SecureString --region ap-south-1

aws ssm put-parameter --name "/skinai/SECRET_KEY" `
  --value "your-super-secret-key-here" `
  --type SecureString --region ap-south-1

aws ssm put-parameter --name "/skinai/CLOUDINARY_CLOUD_NAME" `
  --value "your_cloud_name" `
  --type SecureString --region ap-south-1

aws ssm put-parameter --name "/skinai/CLOUDINARY_API_KEY" `
  --value "your_api_key" `
  --type SecureString --region ap-south-1

aws ssm put-parameter --name "/skinai/CLOUDINARY_API_SECRET" `
  --value "your_api_secret" `
  --type SecureString --region ap-south-1
```

### Step 4.3 — Create ECS Task Definition (Backend)

1. In ECS Console → **Task Definitions** → **Create new task definition**
2. Fill in:
   - **Task definition name**: `skinai-backend-task`
   - **Launch type**: AWS Fargate
   - **CPU**: 1 vCPU
   - **Memory**: 3 GB (TensorFlow needs memory!)
3. Under **Container**:
   - **Image URI**: `YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com/skinai-backend:latest`
   - **Port**: 8000
4. Under **Environment variables** → Add from SSM:
   - `MONGO_URL` → from SSM `/skinai/MONGO_URL`
   - `SECRET_KEY` → from SSM `/skinai/SECRET_KEY`
   - `CLOUDINARY_CLOUD_NAME` → from SSM `/skinai/CLOUDINARY_CLOUD_NAME`
   - `CLOUDINARY_API_KEY` → from SSM `/skinai/CLOUDINARY_API_KEY`
   - `CLOUDINARY_API_SECRET` → from SSM `/skinai/CLOUDINARY_API_SECRET`
   - `FRONTEND_URL` → (your actual Vercel or CloudFront URL, plain text)
5. Click **Create**

### Step 4.4 — Create an ECS Service for Backend

1. Go to your **ECS Cluster** → **Services** → **Create**
2. Fill in:
   - **Launch type**: FARGATE
   - **Task definition**: `skinai-backend-task`
   - **Service name**: `skinai-backend-service`
   - **Desired tasks**: 1 (scale up later)
3. **Networking**:
   - Choose your default VPC
   - Select 2 subnets (different availability zones)
   - Create a new **Security Group**: allow inbound TCP port **8000** from the ALB
4. **Load Balancing**:
   - Create new **Application Load Balancer**
   - **Name**: `skinai-alb`
   - **Listener**: HTTP port 80 → forward to backend on port 8000
5. Click **Create**

> After a few minutes, your backend will be running. Copy the **ALB DNS name** (e.g., `skinai-alb-123456.ap-south-1.elb.amazonaws.com`)

---

## 🌐 PART 5 — Deploy Frontend on S3 + CloudFront

> S3 = file storage | CloudFront = global CDN (super fast delivery worldwide)

### Step 5.1 — Create an S3 Bucket

```powershell
# Create bucket (name must be globally unique)
aws s3api create-bucket `
  --bucket skinai-frontend-prod `
  --region ap-south-1 `
  --create-bucket-configuration LocationConstraint=ap-south-1

# Enable static website hosting
aws s3 website s3://skinai-frontend-prod `
  --index-document index.html `
  --error-document index.html
```

### Step 5.2 — Build Frontend with Real Backend URL

```powershell
cd d:\SkinAi\frontend

# Replace with your actual ALB URL
$env:VITE_API_URL = "http://skinai-alb-123456.ap-south-1.elb.amazonaws.com/api/v1"

npm run build
```

### Step 5.3 — Upload to S3

```powershell
aws s3 sync dist/ s3://skinai-frontend-prod --delete
```

### Step 5.4 — Create CloudFront Distribution

1. AWS Console → **CloudFront** → **Create Distribution**
2. **Origin domain**: Select your S3 bucket
3. **Origin access**: ✅ Origin access control settings (recommended)
4. **Default root object**: `index.html`
5. Under **Error pages** → Add custom error:
   - HTTP Error: `403` → Response page: `/index.html` → HTTP 200
   - HTTP Error: `404` → Response page: `/index.html` → HTTP 200
   (This is needed for React Router to work)
6. Click **Create Distribution**

> Copy the **CloudFront domain** (e.g., `d1234abcd.cloudfront.net`)

---

## 🔒 PART 6 — Add HTTPS / Custom Domain (Optional but Recommended)

### Step 6.1 — Register a Domain (If you don't have one)

- AWS Route 53 → **Registered domains** → **Register domain**
- Example: `skinai.in` costs ~$10/yr

### Step 6.2 — Request SSL Certificate (Free)

```powershell
# Must be in us-east-1 for CloudFront!
aws acm request-certificate `
  --domain-name "skinai.in" `
  --subject-alternative-names "*.skinai.in" `
  --validation-method DNS `
  --region us-east-1
```

Then validate it by adding CNAME records in Route 53 (AWS shows you exactly what to add).

### Step 6.3 — Update CloudFront with Domain + SSL

1. CloudFront → Your distribution → **Edit**
2. **Alternate domain names**: `skinai.in`, `www.skinai.in`
3. **SSL Certificate**: Choose the one you created
4. Save

### Step 6.4 — Point Domain to ALB (for Backend API)

1. Route 53 → **Hosted zones** → your domain
2. Create record:
   - **Name**: `api`
   - **Type**: A → Alias → Application Load Balancer
   - Choose your ALB
3. Now backend is at: `https://api.skinai.in`

### Step 6.5 — Update CORS in Backend

Update `backend/app/main.py` origins list:

```python
origins = [
    "http://localhost:5173",
    "https://skinai.in",
    "https://www.skinai.in",
    "https://d1234abcd.cloudfront.net",  # your CloudFront URL
]
```

Then rebuild and push the backend image again (repeat Part 3 Step 3.3).

---

## 💰 PART 7 — Estimated Cost (AWS Free Tier)

| Service | Free Tier | After Free Tier |
|---------|-----------|-----------------|
| ECS Fargate | 1 vCPU + 3GB, 1 task | ~$30-50/month |
| S3 | 5GB storage, 20K requests | ~$1/month |
| CloudFront | 1TB transfer, 10M requests | ~$1-5/month |
| ALB | 750 hours | ~$20/month |
| ECR | 500MB storage | ~$1/month |
| **Total** | **~$0 for 12 months** | **~$55-80/month** |

> 💡 **Tip**: AWS Free Tier covers 12 months. After that, consider Railway (backend) + Vercel (frontend) which are cheaper for personal projects.

---

## 🔄 PART 8 — Updating Your App (CI/CD Workflow)

Every time you update your code:

```powershell
# === Backend Update ===
cd d:\SkinAi\backend
docker build -t skinai-backend:latest .
docker tag skinai-backend:latest $ECR_BASE/skinai-backend:latest
docker push $ECR_BASE/skinai-backend:latest

# Force ECS to redeploy (pulls new image)
aws ecs update-service `
  --cluster skinai-cluster `
  --service skinai-backend-service `
  --force-new-deployment `
  --region ap-south-1

# === Frontend Update ===
cd d:\SkinAi\frontend
npm run build
aws s3 sync dist/ s3://skinai-frontend-prod --delete

# Invalidate CloudFront cache (so users get new version immediately)
aws cloudfront create-invalidation `
  --distribution-id YOUR_DISTRIBUTION_ID `
  --paths "/*"
```

---

## ✅ Final Checklist

- [ ] Docker Desktop installed and running
- [ ] AWS CLI configured (`aws configure`)
- [ ] `backend/Dockerfile` created
- [ ] `frontend/Dockerfile` created
- [ ] `frontend/nginx.conf` created
- [ ] Both images tested locally with `docker-compose up`
- [ ] ECR repositories created
- [ ] Images pushed to ECR
- [ ] ECS Cluster created (Fargate)
- [ ] Backend secrets stored in SSM Parameter Store
- [ ] ECS Task Definition created
- [ ] ECS Service + ALB running
- [ ] S3 bucket created for frontend
- [ ] Frontend built and uploaded to S3
- [ ] CloudFront distribution created
- [ ] (Optional) Custom domain + SSL configured
- [ ] CORS updated in backend for production URLs

---

## 🆘 Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| Container exits immediately | Check logs: `aws logs tail /ecs/skinai-backend` |
| 502 Bad Gateway | Backend container crashed — check ECS task logs |
| CORS error in browser | Add frontend URL to `origins` list in `main.py` |
| model.h5 not found | Make sure it's copied in Dockerfile (not in .dockerignore) |
| TensorFlow OOM error | Increase task memory to 4GB in ECS task definition |
| React Router 404 on refresh | Ensure CloudFront error pages are set (Step 5.4) |

---

*Guide created for SkinAI project — FastAPI + React/Vite + model.h5 + MongoDB Atlas + Cloudinary*
