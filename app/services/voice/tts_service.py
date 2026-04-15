"""
TTS Service — Kokoro self-hosted for voice note replies.

Kokoro TTS:
  - Best quality self-hosted TTS in 2026
  - ~2GB VRAM / can run on CPU (slower)
  - Apache 2.0 license — fully free
  - Supports multiple voices and styles

Deploy Kokoro:
  docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:v0.2.2
  (GPU version: ghcr.io/remsky/kokoro-fastapi-gpu:v0.2.2)

Voice IDs (Kokoro):
  af_heart  - warm female (default — good for PA contact)
  am_adam   - male voice
  bf_emma   - British female
  bm_george - British male

Falls back to ElevenLabs API if self-hosted Kokoro is unavailable.
"""
import httpx
from app.core.config import settings

DEFAULT_VOICE = "af_heart"


async def synthesise_speech(text: str, voice_id: str | None = None) -> bytes:
    """
    Convert text to speech. Returns MP3 bytes.
    Uses self-hosted Kokoro, falls back to ElevenLabs.
    """
    voice = voice_id or DEFAULT_VOICE

    if settings.KOKORO_SERVICE_URL:
        try:
            return await _synthesise_kokoro(text, voice)
        except Exception:
            pass  # fall through to ElevenLabs

    if settings.ELEVENLABS_API_KEY:
        return await _synthesise_elevenlabs(text, voice)

    raise RuntimeError("No TTS service available — configure KOKORO_SERVICE_URL or ELEVENLABS_API_KEY")


async def _synthesise_kokoro(text: str, voice: str) -> bytes:
    """Call self-hosted Kokoro FastAPI service."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{settings.KOKORO_SERVICE_URL}/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": text,
                "voice": voice,
                "response_format": "mp3",
                "speed": 1.0,
            },
        )
        r.raise_for_status()
        return r.content


async def _synthesise_elevenlabs(text: str, voice_id: str) -> bytes:
    """ElevenLabs fallback — only used if Kokoro is down."""
    elevenlabs_voice = voice_id if len(voice_id) > 10 else "EXAVITQu4vr4xnSDxMaL"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice}/stream",
            headers={"xi-api-key": settings.ELEVENLABS_API_KEY},
            json={
                "text": text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
        )
        r.raise_for_status()
        return r.content
