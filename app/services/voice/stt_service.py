"""
STT Service — multi-provider fallback:

1. ElevenLabs (default)
2. Self-hosted Faster-Whisper (free, if WHISPER_SERVICE_URL is set)
3. OpenAI Whisper API (last resort)
"""
import httpx
import io
from openai import AsyncOpenAI
from app.core.config import settings


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

async def transcribe_audio(file_url: str) -> str:
    """
    Transcribe a voice note (async).

    Priority:
      1. ElevenLabs (default)
      2. Self-hosted Whisper
      3. OpenAI Whisper API
    """

    # 1. ElevenLabs (default)
    if settings.ELEVENLABS_API_KEY:
        try:
            return await _transcribe_elevenlabs(file_url)
        except Exception as e:
            print(f"[STT] ElevenLabs failed: {e}")

    # 2. Self-hosted Whisper (if WHISPER_SERVICE_URL is set)
    if settings.WHISPER_SERVICE_URL:
        try:
            return await _transcribe_self_hosted(file_url)
        except Exception as e:
            print(f"[STT] Self-hosted Whisper failed: {e}")

    # 3. OpenAI fallback
    if settings.OPENAI_API_KEY:
        try:
            return await _transcribe_openai_api(file_url)
        except Exception as e:
            print(f"[STT] OpenAI Whisper failed: {e}")

    raise Exception("No STT provider available")


# ─────────────────────────────────────────────────────────────
# ElevenLabs (DEFAULT)
# ─────────────────────────────────────────────────────────────

async def _transcribe_elevenlabs(file_url: str) -> str:
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.get(file_url)
        r.raise_for_status()

        audio_bytes = r.content

    # run sync SDK safely in thread
    import asyncio

    def _run():
        result = client.speech_to_text.convert(
            file=audio_bytes,
            model_id="scribe_v1",
        )
        return result.text

    return await asyncio.to_thread(_run)


# ─────────────────────────────────────────────────────────────
# Self-hosted Whisper
# ─────────────────────────────────────────────────────────────

async def _transcribe_self_hosted(file_url: str) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        audio_r = await client.get(file_url)
        audio_bytes = audio_r.content

        r = await client.post(
            f"{settings.WHISPER_SERVICE_URL}/v1/audio/transcriptions",
            data={"model": "Systran/faster-whisper-small"},
            files={"file": ("voice.m4a", audio_bytes, "audio/m4a")},
        )

        r.raise_for_status()
        return r.json()["text"]


# ─────────────────────────────────────────────────────────────
# OpenAI fallback
# ─────────────────────────────────────────────────────────────

async def _transcribe_openai_api(file_url: str) -> str:
    oai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async with httpx.AsyncClient() as client:
        r = await client.get(file_url)
        r.raise_for_status()

        audio_file = io.BytesIO(r.content)
        audio_file.name = "voice.m4a"

    result = await oai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )

    return result.text