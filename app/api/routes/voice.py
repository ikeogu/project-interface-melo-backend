"""
Voice upload endpoint.

Client uploads raw audio → backend stores it → returns public URL.
Then client sends a regular message with content_type=voice and the URL.

Storage: Supabase Storage (free, already set up) or S3.
For MVP we use Supabase Storage via the REST API — no extra config needed.
"""
import uuid
import httpx
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from pydantic import BaseModel
from app.api.deps import get_current_user
from app.models.user import User
from app.core.config import settings

router = APIRouter(prefix="/voice", tags=["voice"])


class UploadResponse(BaseModel):
    url: str
    key: str


@router.post("/upload", response_model=UploadResponse)
async def upload_voice_note(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a voice note file.
    Returns a public URL that can be used in a message.

    Storage priority:
      1. Supabase Storage (if SUPABASE_URL is set — already is)
      2. AWS S3 (if AWS keys set)
      3. Local fallback (dev only — not for production)
    """
    audio_bytes = await file.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(audio_bytes) > 25 * 1024 * 1024:  # 25MB max
        raise HTTPException(status_code=400, detail="File too large (max 25MB)")

    key = f"voice-notes/{current_user.id}/{uuid.uuid4()}.m4a"

    # Try Supabase Storage first
    if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY:
        url = await _upload_to_supabase(audio_bytes, key)
        return UploadResponse(url=url, key=key)

    # Try S3
    if settings.AWS_ACCESS_KEY_ID:
        from app.services.voice.storage_service import upload_audio
        url = upload_audio(audio_bytes, folder="voice-notes")
        return UploadResponse(url=url, key=key)

    # Dev fallback — return a placeholder (voice won't actually play)
    if settings.ENVIRONMENT == "development":
        print(f"[DEV] Voice upload received: {len(audio_bytes)} bytes — no storage configured")
        return UploadResponse(url="", key=key)

    raise HTTPException(
        status_code=503,
        detail="No storage configured. Set SUPABASE_URL/SERVICE_KEY or AWS credentials."
    )


async def _upload_to_supabase(audio_bytes: bytes, key: str) -> str:
    """Upload to Supabase Storage and return public URL."""
    bucket = "voice-notes"
    supabase_url = settings.SUPABASE_URL.rstrip("/")

    async with httpx.AsyncClient(timeout=60) as client:
        # Upload the file
        r = await client.post(
            f"{supabase_url}/storage/v1/object/{bucket}/{key}",
            headers={
                "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
                "Content-Type": "audio/m4a",
                "x-upsert": "true",
            },
            content=audio_bytes,
        )

        if r.status_code not in (200, 201):
            # Try to create bucket first if it doesn't exist
            await _ensure_bucket(client, bucket)
            r = await client.post(
                f"{supabase_url}/storage/v1/object/{bucket}/{key}",
                headers={
                    "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
                    "Content-Type": "audio/m4a",
                    "x-upsert": "true",
                },
                content=audio_bytes,
            )
            if r.status_code not in (200, 201):
                raise HTTPException(status_code=500, detail=f"Supabase upload failed: {r.text}")

    # Return public URL
    return f"{supabase_url}/storage/v1/object/public/{bucket}/{key}"


async def _ensure_bucket(client: httpx.AsyncClient, bucket: str):
    """Create Supabase Storage bucket if it doesn't exist."""
    supabase_url = settings.SUPABASE_URL.rstrip("/")
    await client.post(
        f"{supabase_url}/storage/v1/bucket",
        headers={
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
        },
        json={"id": bucket, "name": bucket, "public": True},
    )