"""
LiveKit Agent entrypoint.

This script is run as a separate process alongside the FastAPI backend.
It connects to the LiveKit server, listens for new rooms, and spins up
a CallPipeline for each incoming call.

Room metadata (set by FastAPI when creating the call session) must contain:
  {
    "contact_id": "uuid",
    "user_id": "uuid",
    "contact_name": "Alex",
    "persona_prompt": "You are Alex...",
    "voice_id": "af_heart",
    "memory_context": "- User is a Lagos-based founder..."
  }

Start the agent:
  cd agent/
  python main.py start        # production (connects to LiveKit cloud/self-hosted)
  python main.py dev          # dev mode (auto-creates test room)

Environment variables (same .env as the main app):
  LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
  ANTHROPIC_API_KEY, OPENROUTER_API_KEY
  WHISPER_SERVICE_URL, KOKORO_SERVICE_URL, VOXTRAL_SERVICE_URL
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Load .env from the parent directory
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm as lk_llm,
)
from livekit.agents.pipeline import VoicePipelineAgent

from agent.context import CallContext
from agent.pipeline import CallPipeline
from agent.stt import FasterWhisperSTT
from agent.tts import SelfHostedTTS, voice_for_contact
from agent.llm import CallLLM, LLMConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("agent.main")


def _load_config() -> dict:
    return {
        "livekit_url":      os.getenv("LIVEKIT_URL", "ws://localhost:7880"),
        "livekit_api_key":  os.getenv("LIVEKIT_API_KEY", "devkey"),
        "livekit_secret":   os.getenv("LIVEKIT_API_SECRET", "secret"),
        "anthropic_key":    os.getenv("ANTHROPIC_API_KEY", ""),
        "openrouter_key":   os.getenv("OPENROUTER_API_KEY", ""),
        "whisper_url":      os.getenv("WHISPER_SERVICE_URL", "http://localhost:8001"),
        "kokoro_url":       os.getenv("KOKORO_SERVICE_URL", "http://localhost:8880"),
        "voxtral_url":      os.getenv("VOXTRAL_SERVICE_URL", ""),
        "elevenlabs_key":   os.getenv("ELEVENLABS_API_KEY", ""),
        "app_base_url":     os.getenv("APP_BASE_URL", "http://localhost:8000"),
    }


async def entrypoint(ctx: JobContext):
    """
    Called once per room when a user initiates a call.
    Parses room metadata, builds the pipeline, and starts the agent.
    """
    cfg = _load_config()

    # Parse room metadata — set by FastAPI in calls.py
    try:
        metadata = json.loads(ctx.room.metadata or "{}")
    except json.JSONDecodeError:
        metadata = {}

    contact_id   = metadata.get("contact_id", "unknown")
    user_id      = metadata.get("user_id", "unknown")
    contact_name = metadata.get("contact_name", "Assistant")
    persona      = metadata.get("persona_prompt", "You are a helpful AI assistant.")
    voice_id     = metadata.get("voice_id", "af_heart")
    memory       = metadata.get("memory_context", "")
    specialty    = metadata.get("specialty_tags", [])

    log.info(f"Call starting — room: {ctx.room.name} | contact: {contact_name} | user: {user_id}")

    # Build call context
    call_ctx = CallContext(
        room_name=ctx.room.name,
        user_id=user_id,
        contact_id=contact_id,
        contact_name=contact_name,
        persona_prompt=persona,
        memory_context=memory,
    )

    # Initialise services
    stt = FasterWhisperSTT(service_url=cfg["whisper_url"])

    tts = SelfHostedTTS(
        voxtral_url=cfg["voxtral_url"],
        kokoro_url=cfg["kokoro_url"],
        elevenlabs_key=cfg["elevenlabs_key"],
    )

    llm = CallLLM(config=LLMConfig(
        anthropic_key=cfg["anthropic_key"],
        openrouter_key=cfg["openrouter_key"],
        app_base_url=cfg["app_base_url"],
    ))

    # Connect to room (subscribe to audio only — no video needed for voice calls)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Start pipeline
    pipeline = CallPipeline(ctx, call_ctx, stt, tts, llm)
    await pipeline.start()

    # Keep running until room closes
    await ctx.room.run_until_disconnected()

    # ── Post-call: save transcript and extract memories ─────────────────

    log.info(f"Call ended — {call_ctx.turn_count} turns | saving transcript")
    await _save_call_results(call_ctx, cfg["app_base_url"])


async def _save_call_results(call_ctx: CallContext, api_base_url: str):
    """
    After the call ends, POST the transcript to the main API.
    The API saves it to the chat history and queues memory extraction.
    """
    import httpx
    transcript = call_ctx.get_full_transcript()
    if not transcript:
        return

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(
                f"{api_base_url}/api/v1/calls/transcript",
                json={
                    "user_id": call_ctx.user_id,
                    "contact_id": call_ctx.contact_id,
                    "room_name": call_ctx.room_name,
                    "transcript": transcript,
                    "turn_count": call_ctx.turn_count,
                    "duration_seconds": int(
                        (asyncio.get_event_loop().time() - 0)  # placeholder
                    ),
                },
            )
        log.info(f"Transcript saved for room {call_ctx.room_name}")
    except Exception as e:
        log.error(f"Failed to save transcript: {e}")


def prewarm(proc: JobProcess):
    """
    Pre-warm the agent process before calls arrive.
    Load models into memory so first call has no cold start.
    """
    log.info("Prewarming agent process...")
    # Nothing to preload for HTTP-based services,
    # but if we added local model loading it would go here.


if __name__ == "__main__":
    cfg = _load_config()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            worker_type="room",             # one agent instance per room
            ws_url=cfg["livekit_url"],
            api_key=cfg["livekit_api_key"],
            api_secret=cfg["livekit_secret"],
        )
    )
