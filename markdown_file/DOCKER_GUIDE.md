# 🐳 DermAura Docker Deployment Guide

This guide provides step-by-step instructions to run the entire DermAura application (Frontend & Backend) using Docker and Docker Compose.

## 📋 Prerequisites
- **Docker Desktop** installed and running on your machine.
- An `.env` file in the root directory with necessary API keys (Cloudinary, etc.).

---

## 🚀 Quick Start (Local Deployment)

### 1. Build and Start the Containers
Open your terminal in the project root and run:
```bash
docker-compose up --build
```
This command will:
- Build the **Backend** image (FastAPI + ML Model).
- Build the **Frontend** image (Vite + Nginx production build).
- Start both services and link them.

### 2. Access the Application
- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛠 Troubleshooting

### Port Conflicts
If you see an error saying `Bind for 0.0.0.0:8000 failed`, it means something else is already using that port.
- **Fix**: Stop any existing processes (like `npm run dev` or local `uvicorn`) before running Docker.

### Environment Variables
The Docker setup looks for a `.env` file in the **root** folder. Ensure your Cloudinary and Database URLs are correctly set there.
> [!IMPORTANT]
> Since the frontend is built into Nginx, and the browser communicates with the API, `VITE_API_URL` should remain `http://localhost:8000` for local Docker testing.

### Rebuilding After Changes
If you modify the code and want to see changes in Docker, always use the `--build` flag:
```bash
docker-compose up --build
```

---

## 🧹 Cleanup
To stop the containers and remove them:
```bash
docker-compose down
```
To remove unused Docker images to save space:
```bash
docker image prune -a
```
