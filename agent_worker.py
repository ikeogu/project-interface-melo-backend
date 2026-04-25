"""
LiveKit Voice Agent Worker — runs as a separate Railway service.

When a user starts a call:
  1. The FastAPI app creates a LiveKit room with contact metadata embedded
  2. This worker picks up the new room job automatically
  3. It joins as the AI participant and runs the pipeline:
       Silero VAD → Groq Whisper STT → Claude (via OpenRouter) → ElevenLabs TTS
  4. After the call ends, it posts the transcript to the API for memory extraction

Run locally:
  python agent_worker.py dev

Run in production (Railway):
  python agent_worker.py start
"""
import os
import json
import logging
import asyncio

import httpx
from dotenv import load_dotenv

load_dotenv()

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm as agents_llm,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai as lk_openai
from livekit.plugins import elevenlabs as lk_elevenlabs
from livekit.plugins import silero

logger = logging.getLogger("call-agent")
logging.basicConfig(level=logging.INFO)

API_BASE = os.getenv("APP_BASE_URL", "http://localhost:8000") + "/api/v1"

# ── Style rule appended to every call system prompt ────────────────────────────
_VOICE_RULES = (
    "\n\nThis is a live voice call. Important:\n"
    "- Keep every reply to 1–3 sentences. Never list or enumerate.\n"
    "- Speak naturally and conversationally. No markdown, no headers.\n"
    "- Ask only one question at a time if you need clarification.\n"
    "- If you don't know something, say so simply and move on."
)


async def entrypoint(ctx: JobContext):
    """Entry point — called once per room by the worker."""
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # ── Parse contact metadata embedded in the room ─────────────────────────────
    metadata: dict = {}
    if ctx.room.metadata:
        try:
            metadata = json.loads(ctx.room.metadata)
        except Exception:
            logger.warning("[Agent] Could not parse room metadata")

    contact_name  = metadata.get("contact_name", "Assistant")
    persona_prompt = metadata.get("persona_prompt", "You are a helpful assistant.")
    voice_id       = metadata.get("voice_id") or "EXAVITQu4vr4xnSDxMaL"  # Alex default
    memory_context = metadata.get("memory_context", "")
    user_id        = metadata.get("user_id", "")
    contact_id     = metadata.get("contact_id", "")

    # ── Build system prompt ─────────────────────────────────────────────────────
    system = persona_prompt
    if memory_context:
        system += f"\n\n--- What you know about this user ---\n{memory_context}\n---"
    system += _VOICE_RULES

    initial_ctx = agents_llm.ChatContext().append(role="system", text=system)

    # ── STT: Groq Whisper (OpenAI-compatible endpoint) ──────────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        stt = lk_openai.STT(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            model="whisper-large-v3-turbo",
        )
        logger.info("[Agent] STT: Groq Whisper")
    else:
        # Fall back to OpenAI Whisper
        stt = lk_openai.STT(api_key=os.getenv("OPENAI_API_KEY"), model="whisper-1")
        logger.info("[Agent] STT: OpenAI Whisper (fallback)")

    # ── LLM: Claude via OpenRouter (OpenAI-compatible) ──────────────────────────
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    anthropic_key  = os.getenv("ANTHROPIC_API_KEY")

    if openrouter_key and not openrouter_key.startswith("sk-or-v1-placeholder"):
        llm = lk_openai.LLM(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            model="anthropic/claude-sonnet-4-20250514",
        )
        logger.info("[Agent] LLM: Claude via OpenRouter")
    elif anthropic_key:
        # Direct Claude via OpenAI-compatible shim is not available;
        # fall back to Qwen which IS OpenAI-compatible on OpenRouter
        llm = lk_openai.LLM(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key or "",
            model="qwen/qwen3-235b-a22b",
        )
        logger.info("[Agent] LLM: Qwen via OpenRouter (Claude key set but OpenRouter unavailable)")
    else:
        raise RuntimeError("No LLM API key configured for the call agent")

    # ── TTS: ElevenLabs ─────────────────────────────────────────────────────────
    el_key = os.getenv("ELEVENLABS_API_KEY")
    if el_key:
        tts = lk_elevenlabs.TTS(
            api_key=el_key,
            voice_id=voice_id,
            model="eleven_turbo_v2",
            encoding="mp3_44100_128",
        )
        logger.info(f"[Agent] TTS: ElevenLabs voice {voice_id}")
    else:
        raise RuntimeError("ELEVENLABS_API_KEY required for voice calls")

    # ── VAD ─────────────────────────────────────────────────────────────────────
    vad = silero.VAD.load()

    # ── Build and start the voice assistant ─────────────────────────────────────
    assistant = VoiceAssistant(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        chat_ctx=initial_ctx,
        allow_interruptions=True,
        interrupt_speech_duration=0.6,
        interrupt_min_words=2,
    )
    assistant.start(ctx.room)

    # Greet the user once connected
    await asyncio.sleep(0.5)
    await assistant.say(
        f"Hey, it's {contact_name}. Go ahead — I'm listening.",
        allow_interruptions=True,
    )

    # ── Track transcript for memory extraction ───────────────────────────────────
    transcript_lines: list[str] = []

    @assistant.on("user_speech_committed")
    def on_user(msg: agents_llm.ChatMessage):
        text = msg.content if isinstance(msg.content, str) else ""
        if text:
            transcript_lines.append(f"User: {text}")

    @assistant.on("agent_speech_committed")
    def on_agent(msg: agents_llm.ChatMessage):
        text = msg.content if isinstance(msg.content, str) else ""
        if text:
            transcript_lines.append(f"{contact_name}: {text}")

    # ── Wait until the user hangs up ─────────────────────────────────────────────
    await ctx.wait_for_disconnect()
    logger.info(f"[Agent] Call ended — {len(transcript_lines)} lines of transcript")

    # ── Post transcript to API (saves to DB + triggers memory extraction) ────────
    if transcript_lines and user_id and contact_id:
        transcript = "\n".join(transcript_lines)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{API_BASE}/calls/transcript",
                    json={
                        "user_id": user_id,
                        "contact_id": contact_id,
                        "room_name": ctx.room.name,
                        "transcript": transcript,
                        "turn_count": len(transcript_lines) // 2,
                        "duration_seconds": 0,
                    },
                )
                resp.raise_for_status()
            logger.info("[Agent] Transcript saved successfully")
        except Exception as e:
            logger.error(f"[Agent] Failed to save transcript: {e}")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=os.getenv("LIVEKIT_URL", "ws://localhost:7880"),
            api_key=os.getenv("LIVEKIT_API_KEY", "devkey"),
            api_secret=os.getenv("LIVEKIT_API_SECRET", "secret"),
        )
    )
