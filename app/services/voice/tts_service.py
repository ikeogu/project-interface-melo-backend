"""
TTS Service — multi-provider fallback:

1. Kokoro (self-hosted, best quality — needs KOKORO_SERVICE_URL)
2. edge-tts (free, no API key, Microsoft neural voices)
3. ElevenLabs (fallback, needs ELEVENLABS_API_KEY)
"""
import httpx
from app.core.config import settings

DEFAULT_VOICE = "af_heart"
EDGE_TTS_VOICE = "en-US-JennyNeural"


async def synthesise_speech(text: str, voice_id: str | None = None) -> bytes:
    """Convert text to speech. Returns MP3 bytes."""
    voice = voice_id or DEFAULT_VOICE

    if settings.KOKORO_SERVICE_URL:
        try:
            return await _synthesise_kokoro(text, voice)
        except Exception:
            pass

    try:
        return await _synthesise_edge_tts(text)
    except Exception:
        pass

    if settings.ELEVENLABS_API_KEY:
        return await _synthesise_elevenlabs(text, voice)

    raise RuntimeError("No TTS service available")


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


async def _synthesise_edge_tts(text: str) -> bytes:
    """Microsoft Edge TTS — free, no API key required."""
    import edge_tts
    import io

    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()


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
