"""
Real-time AI voice call pipeline.

Stack:
  WebRTC SFU:    LiveKit (self-hosted)
  STT:           Faster-Whisper (self-hosted, streaming mode)
  LLM:           Claude → Qwen fallback
  TTS:           Voxtral (self-hosted, speech-to-speech OR text-to-speech mode)

Pipeline per audio chunk:
  LiveKit audio track
    → Faster-Whisper (streaming, end-of-turn detection)
    → Claude / Qwen (generate text response)
    → Voxtral (text → speech)
    → Publish back as LiveKit audio track

Latency budget (self-hosted on GPU server):
  Faster-Whisper:  ~150-300ms
  Claude API:      ~400-800ms
  Voxtral TTS:     ~200-400ms
  Total:           ~750ms - 1.5s  ← acceptable for conversational AI

Deploy Voxtral (Mistral's voice model):
  pip install mistral-common vllm
  vllm serve mistralai/Voxtral-Small-22B-v0.1 --port 8890

Deploy Faster-Whisper for streaming:
  pip install faster-whisper
  # or use the whisper-service container in docker-compose.yml
"""
import httpx
import asyncio
from app.core.config import settings
from app.services.ai.claude_service import get_response


async def process_voice_turn(
    audio_bytes: bytes,
    persona_prompt: str,
    conversation_history: list[dict],
    memory_context: str = "",
    message_count: int = 0,
) -> bytes:
    """
    Full pipeline: audio in → audio out.
    Used by the LiveKit Agent to handle a single conversational turn.
    Returns MP3/PCM audio bytes of the AI's response.
    """

    # 1. STT: transcribe what the user said
    transcript = await _transcribe_streaming(audio_bytes)
    if not transcript.strip():
        return b""

    # 2. LLM: generate response text
    response_text = await get_response(
        persona_prompt=persona_prompt,
        conversation_history=conversation_history,
        user_message=transcript,
        memory_context=memory_context,
        message_count=message_count,
    )

    # 3. TTS: convert response to speech
    audio_response = await _voxtral_tts(response_text)

    return audio_response


async def _transcribe_streaming(audio_bytes: bytes) -> str:
    """
    Faster-Whisper streaming transcription.
    Self-hosted service running in the whisper-service container.
    """
    if not settings.WHISPER_SERVICE_URL:
        return ""

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{settings.WHISPER_SERVICE_URL}/transcribe",
            files={"audio": ("chunk.webm", audio_bytes, "audio/webm")},
            data={"mode": "streaming"},  # faster mode for real-time
        )
        r.raise_for_status()
        return r.json().get("text", "")


async def _voxtral_tts(text: str) -> bytes:
    """
    Voxtral text-to-speech via self-hosted vLLM service.
    Voxtral is speech-to-speech but also supports text-to-speech mode.
    """
    if not settings.VOXTRAL_SERVICE_URL:
        # Fall back to Kokoro for TTS
        from app.services.voice.tts_service import synthesise_speech
        return await synthesise_speech(text)

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{settings.VOXTRAL_SERVICE_URL}/v1/audio/speech",
            json={
                "model": "voxtral",
                "input": text,
                "voice": "default",
                "response_format": "pcm",  # PCM for lowest latency in calls
            },
        )
        r.raise_for_status()
        return r.content
