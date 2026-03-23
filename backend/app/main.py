from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse
import time
import logging
from datetime import datetime

# ── Structured Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("skinai")

# ── Rate Limiter ──
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(title="SkinCare AI API", version="1.0.0")

# Attach limiter to app
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# ── Custom rate limit error handler ──
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down and try again shortly."}
    )

# ── Gzip Compression ──
app.add_middleware(GZipMiddleware, minimum_size=500)

# ── Response Time Middleware ──
@app.middleware("http")
async def log_response_time(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
    if elapsed > 0.5 or request.url.path.startswith("/api"):
        logger.info(f"{request.method} {request.url.path} → {response.status_code} in {elapsed:.2f}s")
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

# ── Startup: pre-load ML model ──
@app.on_event("startup")
async def startup_event():
    logger.info("MongoDB configured")
    try:
        from app.services.ml_model import _get_model
        logger.info("Pre-loading ML model...")
        _get_model()
        logger.info("ML model ready!")
    except Exception as e:
        logger.warning(f"Model pre-load failed (will lazy-load): {e}")

# ── Shutdown: close DB connection ──
@app.on_event("shutdown")
async def shutdown_event():
    from app.core.database import db
    if db.client:
        db.client.close()
        logger.info("MongoDB connection closed")

# ── Health Check ──
@app.get("/health")
async def health_check():
    """Health check for load balancers and monitoring."""
    from app.core.database import db
    try:
        if db.client:
            await db.client.admin.command("ping")
            db_status = "connected"
        else:
            db_status = "not_initialized"
    except Exception:
        db_status = "disconnected"

    from app.services.ml_model import _model
    model_status = "loaded" if _model is not None else "not_loaded"

    return {
        "status": "healthy",
        "database": db_status,
        "ml_model": model_status,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.get("/")
async def root():
    return {"message": "Welcome to SkinCare AI API"}
