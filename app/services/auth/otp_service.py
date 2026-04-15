"""
OTP Service — stores codes in memory for dev, Redis for production.

In development (ENVIRONMENT=development):
  - OTPs stored in a simple dict in memory
  - Code always printed to terminal
  - No Redis needed

In production:
  - OTPs stored in Redis with TTL
  - Requires a working REDIS_URL
"""
import random
import logging
logger = logging.getLogger(__name__)

from datetime import datetime, timedelta
from app.core.config import settings
from app.services.auth.email_service import send_otp_email

OTP_TTL_SECONDS = 600
MAX_ATTEMPTS = 5

# In-memory store for dev — {email: {code, expires_at, attempts}}
_dev_store: dict = {}


def _generate_otp() -> str:
    return str(random.randint(100000, 999999))


async def create_otp(email: str) -> str:
    code = _generate_otp()
    email = email.lower()

    logger.info(f"ENV={settings.ENVIRONMENT}, REDIS_URL={bool(settings.REDIS_URL)}")

    if settings.ENVIRONMENT == "development" or not settings.REDIS_URL:
        _dev_store[email] = {
            "code": code,
            "expires_at": datetime.utcnow() + timedelta(seconds=OTP_TTL_SECONDS),
            "attempts": 0,
        }
        print(f"OTP for {email}: {code}")
        return code

    try:
        await _create_otp_redis(email, code)
    except Exception as e:
        logger.error(f"Redis error: {str(e)}")
        raise

    # SEND OTP HERE
    sent = await send_otp_email(email, code)

    if not sent:
        logger.error(f"OTP email failed for {email}")
        raise Exception("Unable to send OTP email")

    


async def verify_otp(email: str, code: str) -> tuple[bool, str]:
    """Verify OTP. Returns (is_valid, error_message)."""
    email = email.lower()

    if settings.ENVIRONMENT == "development" or not settings.REDIS_URL:
        return _verify_otp_memory(email, code)

    return await _verify_otp_redis(email, code)


# ── In-memory (dev) ────────────────────────────────────────────────────────────

def _verify_otp_memory(email: str, code: str) -> tuple[bool, str]:
    entry = _dev_store.get(email)

    if not entry:
        return False, "Code not found. Request a new one."

    if datetime.utcnow() > entry["expires_at"]:
        del _dev_store[email]
        return False, "Code expired. Request a new one."

    if entry["attempts"] >= MAX_ATTEMPTS:
        return False, "Too many attempts. Request a new code."

    if entry["code"] != code.strip():
        entry["attempts"] += 1
        remaining = MAX_ATTEMPTS - entry["attempts"]
        return False, f"Invalid code. {remaining} attempts remaining."

    del _dev_store[email]
    return True, ""


# ── Redis (production) ─────────────────────────────────────────────────────────

async def _get_redis():
    """Get Redis client — only imported when actually needed."""
    import redis.asyncio as aioredis
    return aioredis.from_url(settings.REDIS_URL, decode_responses=True)


async def _create_otp_redis(email: str, code: str) -> str:
    r = await _get_redis()
    key = f"otp:{email}"
    attempt_key = f"otp_attempts:{email}"
    await r.setex(key, OTP_TTL_SECONDS, code)
    await r.delete(attempt_key)
    return code


async def _verify_otp_redis(email: str, code: str) -> tuple[bool, str]:
    r = await _get_redis()
    key = f"otp:{email}"
    attempt_key = f"otp_attempts:{email}"

    attempts = await r.get(attempt_key)
    if attempts and int(attempts) >= MAX_ATTEMPTS:
        return False, "Too many attempts. Request a new code."

    stored = await r.get(key)
    if not stored:
        return False, "Code expired or not found. Request a new one."

    if stored != code.strip():
        await r.incr(attempt_key)
        await r.expire(attempt_key, OTP_TTL_SECONDS)
        remaining = MAX_ATTEMPTS - int(await r.get(attempt_key) or 1)
        return False, f"Invalid code. {remaining} attempts remaining."

    await r.delete(key)
    await r.delete(attempt_key)
    return True, ""