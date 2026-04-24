import uuid
import asyncio
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from pydantic import BaseModel
from app.api.deps import get_current_user
from app.models.user import User
from app.core.config import settings

router = APIRouter(prefix="/voice", tags=["voice"])


class UploadResponse(BaseModel):
    url: str
    key: str
    transcript: str | None = None


@router.post("/upload", response_model=UploadResponse)
async def upload_voice_note(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    audio_bytes = await file.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 25MB)")

    key = f"voice-notes/{current_user.id}/{uuid.uuid4()}.m4a"

    # Transcribe with ElevenLabs (optional)
    transcript = None
    if settings.ELEVENLABS_API_KEY:
        try:
            from app.services.ai.elevenlabs_service import ElevenLabsService
            transcript = await asyncio.to_thread(
                ElevenLabsService().transcribe_bytes,
                audio_bytes
            )
        except Exception as e:
            print(f"[ElevenLabs] transcription failed: {e}")

    from app.services.voice.storage_service import upload_audio
    try:
        url = await upload_audio(audio_bytes, folder="voice-notes", content_type="audio/m4a")
    except RuntimeError:
        if settings.ENVIRONMENT == "development":
            url = ""
        else:
            raise HTTPException(status_code=503, detail="No storage configured")

    return UploadResponse(url=url, key=key, transcript=transcript)

