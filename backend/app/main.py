from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SkinCare AI API", version="1.0.0")

@app.middleware("http")
async def log_requests(request, call_next):
    print(f"DEBUG: Request: {request.method} {request.url}")
    print(f"DEBUG: Origin: {request.headers.get('origin')}")
    response = await call_next(request)
    print(f"DEBUG: Response Status: {response.status_code}")
    return response

# CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.main import api_router
app.include_router(api_router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    from app.core.config import settings
    print(f"DEBUG: LOADED MONGO_URL: {settings.MONGO_URL}")

@app.get("/")
async def root():
    return {"message": "Welcome to SkinCare AI API"}
