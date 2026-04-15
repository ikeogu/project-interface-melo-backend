"""
STT Service — two modes:

1. Async (voice notes):  standard Whisper small, self-hosted via faster-whisper
2. Real-time (calls):    Faster-Whisper streaming, also self-hosted

Self-hosted faster-whisper runs on:
  - CPU-only server: ~2-4s latency (fine for voice notes)
  - GPU server:      ~200-400ms (needed for real-time calls)

Deploy with:
  pip install faster-whisper
  # or via our Docker service defined in docker-compose.yml
"""
import httpx
from app.core.config import settings


async def transcribe_audio(file_url: str) -> str:
    """
    Transcribe a voice note (async, not real-time).
    Calls the self-hosted Whisper service.
    Falls back to OpenAI Whisper API if self-hosted is unavailable.
    """
    # Try self-hosted first
    if settings.WHISPER_SERVICE_URL:
        try:
            return await _transcribe_self_hosted(file_url)
        except Exception:
            pass  # fall through to API fallback

    # Fallback: OpenAI Whisper API (~$0.006/min, only used if self-hosted is down)
    return await _transcribe_openai_api(file_url)


async def _transcribe_self_hosted(file_url: str) -> str:
    """
    Call the self-hosted Faster-Whisper HTTP service.
    This service is defined in docker-compose.yml (whisper-service container).
    """
    import httpx as _httpx
    async with _httpx.AsyncClient(timeout=60) as client:
        # Download audio from S3 first
        audio_r = await client.get(file_url)
        audio_bytes = audio_r.content

        # Send to local Whisper service
        r = await client.post(
            f"{settings.WHISPER_SERVICE_URL}/transcribe",
            files={"audio": ("voice.m4a", audio_bytes, "audio/m4a")},
        )
        r.raise_for_status()
        return r.json()["text"]


async def _transcribe_openai_api(file_url: str) -> str:
    """OpenAI Whisper API fallback."""
    from openai import AsyncOpenAI
    import io

    oai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async with httpx.AsyncClient() as client:
        r = await client.get(file_url)
        audio_file = io.BytesIO(r.content)
        audio_file.name = "voice.m4a"

    transcript = await oai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    return transcript.text
