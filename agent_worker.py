"""
LiveKit Voice Agent Worker — runs as a separate Railway service.

When a user starts a call:
  1. FastAPI creates a LiveKit room with contact metadata in room.metadata
  2. This worker picks up the room job automatically via LiveKit Workers
  3. It joins as the AI participant and runs:
       Silero VAD → Groq Whisper STT → Claude via OpenRouter → ElevenLabs TTS
  4. After the call ends, posts the transcript to /calls/transcript
     so the main API can save it and extract memories

Run locally:
  python agent_worker.py dev

Run in production (Railway agent service):
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
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    TurnHandlingOptions,
)
from livekit.plugins import openai as lk_openai
from livekit.plugins import silero
from edge_tts_plugin import EdgeTTS

logger = logging.getLogger("call-agent")
logging.basicConfig(level=logging.INFO)

API_BASE = os.getenv("APP_BASE_URL", "http://localhost:8000") + "/api/v1"

_VOICE_RULES = """

You are on a live phone call right now. Speak exactly the way a real person does on the phone.

Natural call behaviour:
- Match your energy to the caller's. If they're brief, be brief. If they want to talk, engage more.
- Respond to what was actually said, not what you expected them to say.
- Use natural spoken language — contractions, short sentences, occasional filler ("right", "sure", "got it") but don't overdo it.
- Never list things out loud. If you need to cover multiple points, weave them into natural sentences.
- One thought at a time. Finish a point, then pause (let them respond). Don't monologue.
- If you didn't catch something or need a moment, say so ("sorry, say that again?" or "give me a second").
- No "Certainly!", "Absolutely!", "Great question!" — these sound robotic on a call. Just respond naturally.
- Silence is fine. You don't need to fill every gap.
- Keep replies tight: 1–3 sentences unless they're clearly asking for depth.
- End responses in a way that invites the conversation forward — a natural pause, a soft question, or simply finishing your thought."""


class ContactAgent(Agent):
    def __init__(self, *, instructions: str, contact_name: str):
        super().__init__(instructions=instructions)
        self._contact_name = contact_name

    async def on_enter(self) -> None:
        await self.session.say(
            f"Hey! What's up?",
        )


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # ── Parse contact metadata embedded in the room ─────────────────────────────
    metadata: dict = {}
    if ctx.room.metadata:
        try:
            metadata = json.loads(ctx.room.metadata)
        except Exception:
            logger.warning("[Agent] Could not parse room metadata — using defaults")

    contact_name   = metadata.get("contact_name", "Assistant")
    persona_prompt = metadata.get("persona_prompt", "You are a helpful assistant.")
    voice_id       = metadata.get("voice_id") or "EXAVITQu4vr4xnSDxMaL"
    memory_context = metadata.get("memory_context", "")
    user_id        = metadata.get("user_id", "")
    contact_id     = metadata.get("contact_id", "")

    # ── Build system prompt ─────────────────────────────────────────────────────
    user_name = metadata.get("user_name", "")
    instructions = persona_prompt
    if user_name:
        instructions += f"\n\nThe person calling is called {user_name}. Use their name naturally — not in every sentence, just the way a real person would."
    if memory_context:
        instructions += (
            f"\n\n--- Background context ---\n"
            f"{memory_context}\n"
            f"---\n"
            f"This is a fresh call. Start the conversation naturally without referencing "
            f"any of the above. Only bring up past information if the user raises it first "
            f"or it becomes directly relevant to what they are asking."
        )
    instructions += _VOICE_RULES

    # ── STT: Groq Whisper ───────────────────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        stt = lk_openai.STT(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            model="whisper-large-v3-turbo",
        )
        logger.info("[Agent] STT: Groq Whisper")
    else:
        stt = lk_openai.STT(api_key=os.getenv("OPENAI_API_KEY", ""), model="whisper-1")
        logger.info("[Agent] STT: OpenAI Whisper (fallback)")

    # ── LLM: Groq (fast, free tier, same key as STT) ────────────────────────────
    groq_llm_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
    llm_base_url = "https://api.groq.com/openai/v1" if os.getenv("GROQ_API_KEY") else "https://openrouter.ai/api/v1"
    llm_model = "llama-3.1-8b-instant" if os.getenv("GROQ_API_KEY") else "mistralai/mistral-7b-instruct:free"
    llm = lk_openai.LLM(
        base_url=llm_base_url,
        api_key=groq_llm_key,
        model=llm_model,
    )
    logger.info(f"[Agent] LLM: {llm_model}")

    # ── TTS: edge-tts (Microsoft, free, no API key needed) ─────────────────────
    tts = EdgeTTS(voice_id=voice_id)
    logger.info(f"[Agent] TTS: edge-tts voice_id={voice_id}")

    # ── VAD ─────────────────────────────────────────────────────────────────────
    vad = silero.VAD.load()

    transcript_lines: list[str] = []

    # ── Wire up the session and agent ───────────────────────────────────────────
    session = AgentSession(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        turn_handling=TurnHandlingOptions(allow_interruptions=True),
    )

    # Capture both sides of the conversation via conversation_item_added
    @session.on("conversation_item_added")
    def _on_item(ev):
        msg = ev.item
        text = msg.text_content if hasattr(msg, "text_content") else None
        if not text:
            return
        role = getattr(msg, "role", None)
        if role and str(role) in ("user", "ChatRole.user"):
            transcript_lines.append(f"User: {text}")
        elif role and str(role) in ("assistant", "ChatRole.assistant"):
            transcript_lines.append(f"{contact_name}: {text}")

    agent = ContactAgent(instructions=instructions, contact_name=contact_name)

    await session.start(agent=agent, room=ctx.room)

    # Wait until the session goes quiet (user hung up or connection dropped)
    await session.wait_for_inactive()
    logger.info(f"[Agent] Call ended — {len(transcript_lines)} transcript lines")

    # ── Post transcript to API ───────────────────────────────────────────────────
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
            logger.info("[Agent] Transcript saved")
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
