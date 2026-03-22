"""
Shared constants for the SkinAi backend.
Central place for all configurable values — avoids magic numbers scattered across files.
"""

# ── Upload Limits ──
DAILY_SCAN_LIMIT = 5
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

# ── ML Confidence ──
MIN_CONFIDENCE_THRESHOLD = 0.35

# ── Auth ──
OTP_EXPIRY_MINUTES = 10
JWT_EXPIRY_DAYS = 1
LOGIN_MAX_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 300  # 5 minutes

# ── Password Policy ──
PASSWORD_MIN_LENGTH = 8
