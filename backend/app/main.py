from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI(title="SkinCare AI API", version="1.0.0")

# ── #7: Response Time Middleware ──
@app.middleware("http")
async def log_response_time(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
    # Only log slow requests (> 0.5s) or API calls (skip static assets)
    if elapsed > 0.5 or request.url.path.startswith("/api"):
        print(f"[PERF] {request.method} {request.url.path} → {response.status_code} in {elapsed:.2f}s")
    return response

from app.core.config import settings

# CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://10.99.173.187:5173",
    "http://10.125.164.187:5173",
]

if hasattr(settings, "FRONTEND_URL") and settings.FRONTEND_URL:
    origins.append(settings.FRONTEND_URL)
    if settings.FRONTEND_URL.endswith('/'):
        origins.append(settings.FRONTEND_URL[:-1])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.main import api_router
app.include_router(api_router, prefix="/api/v1")

# ── #1 bonus: Eager model load at startup ──
@app.on_event("startup")
async def startup_event():
    from app.core.config import settings
    print(f"[STARTUP] MongoDB configured")
    # Pre-load the ML model so the first request is fast
    try:
        from app.services.ml_model import _get_model
        print("[STARTUP] Pre-loading ML model...")
        _get_model()
        print("[STARTUP] ML model ready!")
    except Exception as e:
        print(f"[STARTUP] Model pre-load failed (will lazy-load): {e}")

@app.get("/")
async def root():
    return {"message": "Welcome to SkinCare AI API"}
