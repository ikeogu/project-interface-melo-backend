"""
TTS plugin for LiveKit Agents.

Priority order:
  1. Voxtral (self-hosted, GPU) — lowest latency for calls
  2. Kokoro  (self-hosted, CPU) — fallback, very good quality
  3. ElevenLabs (API)           — last resort if both self-hosted are down

Each contact has a voice_id that maps to a Kokoro or Voxtral voice name.
The mapping table below lets each AI contact sound distinct.

Kokoro voices:
  af_heart  — warm female (good for PA, wellness coach)
  am_adam   — calm male
  bf_emma   — British female (good for legal advisor)
  bm_george — British male (good for business analyst)
  af_sky    — energetic female
  am_michael — deep male

Voxtral voices (when GPU available):
  Uses same IDs — Voxtral accepts named voices
"""
import httpx
import asyncio
from typing import AsyncGenerator

# Default voice per contact type (matched by specialty_tags)
VOICE_MAP = {
    "af_heart":  ["personal assistant", "wellness", "pastoral"],
    "bm_george": ["business", "finance", "analyst"],
    "bf_emma":   ["legal", "compliance", "advisor"],
    "am_adam":   ["default"],
}

CHUNK_SIZE = 4096  # stream audio in 4KB chunks for low latency


class SelfHostedTTS:
    """
    TTS plugin for the LiveKit agent pipeline.
    Tries Voxtral first, falls back to Kokoro, then ElevenLabs.
    """

    def __init__(
        self,
        voxtral_url: str = "",
        kokoro_url: str = "",
        elevenlabs_key: str = "",
    ):
        self.voxtral_url = voxtral_url.rstrip("/") if voxtral_url else ""
        self.kokoro_url = kokoro_url.rstrip("/") if kokoro_url else ""
        self.elevenlabs_key = elevenlabs_key

    async def synthesise(self, text: str, voice_id: str = "af_heart") -> bytes:
        """
        Convert text to speech. Returns complete MP3/PCM bytes.
        Used for short responses where streaming isn't needed.
        """
        # Try Voxtral (GPU, lowest latency)
        if self.voxtral_url:
            try:
                return await self._voxtral(text, voice_id)
            except Exception as e:
                print(f"[TTS] Voxtral failed: {e} — falling back to Kokoro")

        # Try Kokoro (CPU, good quality)
        if self.kokoro_url:
            try:
                return await self._kokoro(text, voice_id)
            except Exception as e:
                print(f"[TTS] Kokoro failed: {e} — falling back to ElevenLabs")

        # Last resort: ElevenLabs API
        if self.elevenlabs_key:
            return await self._elevenlabs(text, voice_id)

        raise RuntimeError("No TTS backend available")

    async def stream(
        self, text: str, voice_id: str = "af_heart"
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks as they're generated.
        Lower perceived latency — LiveKit can start playing before full audio is ready.
        """
        if self.voxtral_url:
            try:
                async for chunk in self._voxtral_stream(text, voice_id):
                    yield chunk
                return
            except Exception as e:
                print(f"[TTS] Voxtral stream failed: {e}")

        if self.kokoro_url:
            try:
                async for chunk in self._kokoro_stream(text, voice_id):
                    yield chunk
                return
            except Exception as e:
                print(f"[TTS] Kokoro stream failed: {e}")

        # ElevenLabs doesn't support streaming in self-hosted mode
        # — return full audio as single chunk
        if self.elevenlabs_key:
            audio = await self._elevenlabs(text, voice_id)
            yield audio

    # ── Voxtral ────────────────────────────────────────────────────────

    async def _voxtral(self, text: str, voice: str) -> bytes:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                f"{self.voxtral_url}/v1/audio/speech",
                json={
                    "model": "voxtral",
                    "input": text,
                    "voice": voice,
                    "response_format": "pcm",
                    "sample_rate": 16000,
                },
            )
            r.raise_for_status()
            return r.content

    async def _voxtral_stream(self, text: str, voice: str) -> AsyncGenerator[bytes, None]:
        async with httpx.AsyncClient(timeout=20) as client:
            async with client.stream(
                "POST",
                f"{self.voxtral_url}/v1/audio/speech",
                json={
                    "model": "voxtral",
                    "input": text,
                    "voice": voice,
                    "response_format": "pcm",
                    "stream": True,
                },
            ) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes(CHUNK_SIZE):
                    yield chunk

    # ── Kokoro ─────────────────────────────────────────────────────────

    async def _kokoro(self, text: str, voice: str) -> bytes:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{self.kokoro_url}/v1/audio/speech",
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

    async def _kokoro_stream(self, text: str, voice: str) -> AsyncGenerator[bytes, None]:
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream(
                "POST",
                f"{self.kokoro_url}/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": text,
                    "voice": voice,
                    "response_format": "mp3",
                    "stream": True,
                },
            ) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes(CHUNK_SIZE):
                    yield chunk

    # ── ElevenLabs fallback ────────────────────────────────────────────

    async def _elevenlabs(self, text: str, voice_id: str) -> bytes:
        # Map Kokoro voice names to ElevenLabs IDs
        el_voice = "EXAVITQu4vr4xnSDxMaL"  # default Sarah
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{el_voice}",
                headers={"xi-api-key": self.elevenlabs_key},
                json={
                    "text": text,
                    "model_id": "eleven_turbo_v2_5",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                },
            )
            r.raise_for_status()
            return r.content


def voice_for_contact(specialty_tags: list[str]) -> str:
    """Pick the best Kokoro/Voxtral voice based on contact specialty."""
    tags_lower = [t.lower() for t in specialty_tags]
    for voice, keywords in VOICE_MAP.items():
        if any(kw in tag for tag in tags_lower for kw in keywords):
            return voice
    return "af_heart"  # default
