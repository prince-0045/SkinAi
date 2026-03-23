"""
Tests for authentication and security features.
"""
import pytest
from app.api.routes.auth import validate_password, generate_otp, _login_attempts
from fastapi import HTTPException


class TestPasswordValidation:
    """Tests for password strength validation."""

    def test_short_password_rejected(self):
        """Password shorter than 8 chars must be rejected."""
        with pytest.raises(HTTPException) as exc:
            validate_password("Abc1")
        assert "8 characters" in str(exc.value.detail)

    def test_no_uppercase_rejected(self):
        """Password without uppercase must be rejected."""
        with pytest.raises(HTTPException) as exc:
            validate_password("abcdefg1")
        assert "uppercase" in str(exc.value.detail)

    def test_no_number_rejected(self):
        """Password without number must be rejected."""
        with pytest.raises(HTTPException) as exc:
            validate_password("Abcdefgh")
        assert "number" in str(exc.value.detail)

    def test_valid_password_accepted(self):
        """Valid password should not raise."""
        validate_password("StrongPass1")  # Should not raise

    def test_strong_password_accepted(self):
        """Complex password should pass."""
        validate_password("MyP@ssw0rd!")  # Should not raise


class TestOTPGeneration:
    """Tests for OTP generation."""

    def test_otp_is_6_digits(self):
        """OTP must be exactly 6 digit characters."""
        otp = generate_otp()
        assert len(otp) == 6
        assert otp.isdigit()

    def test_otp_uniqueness(self):
        """Multiple OTPs should generally be different."""
        otps = {generate_otp() for _ in range(100)}
        assert len(otps) > 50  # At least 50 unique out of 100


class TestBruteForceProtection:
    """Tests for login attempt tracking."""

    def test_login_attempts_dict_exists(self):
        """Login attempts tracker must be a defaultdict."""
        from collections import defaultdict
        assert isinstance(_login_attempts, defaultdict)
