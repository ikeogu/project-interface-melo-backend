#!/usr/bin/env python3
"""
POC Terminal Script — mirrors the production claude_service.py exactly.

Production routing (matches app/services/ai/claude_service.py):
  - Messages 1-20:  Claude (primary, best quality)
  - Messages 21+:   Qwen via OpenRouter (cost fallback)
  - Dev/no credits: Ollama (local, free)

This POC lets you test the same routing logic the real API uses,
so you know exactly what users will experience.

.env keys:
  ANTHROPIC_API_KEY=sk-ant-...          ← primary (top up at console.anthropic.com)
  OPENROUTER_API_KEY=sk-or-...          ← fallback after 20 msgs (openrouter.ai)
  OLLAMA_BASE_URL=http://localhost:11434 ← dev fallback (free, local)
  OLLAMA_MODEL=llama3.2
  ELEVENLABS_API_KEY=...                ← voice output (optional)
  ACTIVE_PROVIDER=claude                ← force a provider (optional)
  CLAUDE_MESSAGE_LIMIT=20               ← override switch threshold (optional)
"""

import asyncio
import json
import os
import subprocess
import tempfile

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Config — mirrors app/core/config.py ───────────────────────────────────────
ANTHROPIC_KEY       = os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_KEY      = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_KEY          = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_KEY      = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE    = "EXAVITQu4vr4xnSDxMaL"

OLLAMA_URL          = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "llama3.2")

# Mirrors CLAUDE_MESSAGE_LIMIT in claude_service.py
CLAUDE_MESSAGE_LIMIT = int(os.getenv("CLAUDE_MESSAGE_LIMIT", "20"))

# Force a specific provider regardless of message count (for testing)
# Values: claude | openrouter | ollama | openai | auto
# "auto" = production routing (Claude → Qwen after limit)
ACTIVE_PROVIDER     = os.getenv("ACTIVE_PROVIDER", "auto").lower()

# Mirrors CLAUDE_MODEL in app/core/config.py
CLAUDE_MODEL        = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# Mirrors QWEN_MODEL in claude_service.py
# Paid model — same as production. For free dev use, set OPENROUTER_MODEL in .env
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-235b-a22b")

PERSONA = """You are Alex, the user's dedicated personal assistant.
You are organised, proactive, warm and discreet. You help them manage their time,
research topics, draft communications and stay on top of commitments.
You respond conversationally, like a trusted human assistant would.
Keep responses concise unless detail is requested.
Never mention that you are an AI unless directly asked."""

history: list[dict] = []
message_count: int = 0


# ── Providers — mirrors app/services/ai/claude_service.py ────────────────────

async def via_claude(message: str) -> str:
    """Primary provider. Used for first CLAUDE_MESSAGE_LIMIT messages."""
    if not ANTHROPIC_KEY:
        raise Exception("ANTHROPIC_API_KEY not set — top up at console.anthropic.com")

    import anthropic
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_KEY)

    r = await client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        system=PERSONA,
        messages=history + [{"role": "user", "content": message}],
    )
    return r.content[0].text.strip()


async def via_openrouter(message: str) -> str:
    """
    Cost fallback after CLAUDE_MESSAGE_LIMIT messages.
    Mirrors _get_qwen_response() in claude_service.py.
    Default model: qwen/qwen3-235b-a22b (paid, same as production).
    For free dev testing set OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
    """
    if not OPENROUTER_KEY:
        raise Exception("OPENROUTER_API_KEY not set")

    messages = [{"role": "system", "content": PERSONA}] + history + [{"role": "user", "content": message}]

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "HTTP-Referer": "http://localhost",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": messages,
                "max_tokens": 512,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


async def via_ollama(message: str) -> str:
    """
    Local dev fallback — free, no API key needed.
    Not used in production. Only here for dev when credits run out.
    Requires: ollama serve && ollama pull llama3.2
    """
    if "ollama.com" in OLLAMA_URL:
        raise Exception(
            "OLLAMA_BASE_URL points to ollama.com (website). "
            "Use http://localhost:11434 for local Ollama."
        )

    messages = [{"role": "system", "content": PERSONA}] + history + [{"role": "user", "content": message}]

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


async def via_openai(message: str) -> str:
    """Extra fallback — not used in production stack."""
    if not OPENAI_KEY:
        raise Exception("OPENAI_API_KEY not set")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=OPENAI_KEY)
    r = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": PERSONA}] + history + [{"role": "user", "content": message}],
        max_tokens=512,
    )
    return r.choices[0].message.content.strip()


# ── Routing — mirrors claude_service.py logic ──────────────────────────────────

def _pick_production_provider() -> list[tuple[str, str]]:
    """
    Mirrors _should_use_fallback() in claude_service.py.
    Claude for first N messages, Qwen after that.
    Falls back to Ollama if both are unavailable (dev only).
    Returns ordered list of (label, provider_name) to try.
    """
    use_qwen = message_count >= CLAUDE_MESSAGE_LIMIT

    if use_qwen:
        # Mirror production: Qwen is primary after limit
        order = ["openrouter", "ollama", "openai"]
        reason = f"msg {message_count}/{CLAUDE_MESSAGE_LIMIT} → switched to Qwen (cost saving)"
    else:
        # Mirror production: Claude is primary
        order = ["claude", "ollama", "openrouter", "openai"]
        reason = f"msg {message_count + 1}/{CLAUDE_MESSAGE_LIMIT} → Claude (primary)"

    return order, reason


async def get_response(message: str) -> tuple[str, str, str]:
    """
    Returns (reply, provider_name, routing_reason).
    Routing mirrors production claude_service.py exactly when ACTIVE_PROVIDER=auto.
    """
    all_providers = {
        "claude":      ("Claude",      via_claude),
        "openrouter":  ("OpenRouter",  via_openrouter),
        "ollama":      ("Ollama",      via_ollama),
        "openai":      ("OpenAI",      via_openai),
    }

    def is_configured(name: str) -> bool:
        if name == "claude":      return bool(ANTHROPIC_KEY)
        if name == "openrouter":  return bool(OPENROUTER_KEY)
        if name == "ollama":      return "ollama.com" not in OLLAMA_URL
        if name == "openai":      return bool(OPENAI_KEY)
        return False

    # Determine order
    if ACTIVE_PROVIDER == "auto":
        order, reason = _pick_production_provider()
    else:
        # Manual override — forced provider first, rest as fallback
        order = [ACTIVE_PROVIDER] + [p for p in all_providers if p != ACTIVE_PROVIDER]
        reason = f"forced ({ACTIVE_PROVIDER})"

    providers = [
        (all_providers[p][0], all_providers[p][1])
        for p in order
        if p in all_providers and is_configured(p)
    ]

    if not providers:
        raise Exception(
            "\nNo providers configured. Add to your .env:\n"
            "  ANTHROPIC_API_KEY=...   ← Claude (primary)\n"
            "  OPENROUTER_API_KEY=...  ← Qwen fallback\n"
            "  OLLAMA_BASE_URL=http://localhost:11434  ← free local dev\n"
        )

    errors = []
    for label, fn in providers:
        try:
            reply = await fn(message)
            return reply, label, reason
        except Exception as e:
            errors.append(f"  {label}: {e}")

    raise Exception("All providers failed:\n" + "\n".join(errors))


# ── Voice output ───────────────────────────────────────────────────────────────

async def speak(text: str) -> None:
    if not ELEVENLABS_KEY:
        return

    print("[Speaking...]", end="", flush=True)
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE}",
            headers={"xi-api-key": ELEVENLABS_KEY, "Content-Type": "application/json"},
            json={
                "text": text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
        )
        r.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(r.content)
        tmp_path = f.name

    for player in ["afplay", "mpg123", "ffplay -nodisp -autoexit"]:
        try:
            subprocess.run(player.split() + [tmp_path], check=True,
                           capture_output=True, timeout=30)
            break
        except (subprocess.SubprocessError, FileNotFoundError):
            continue

    print(" done")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    global message_count

    print("=" * 56)
    print("  AI Contacts — POC (mirrors production routing)")
    print("  Talking to: Alex (Personal Assistant)")
    print("=" * 56)
    print()

    # Show config
    routing = "auto (Claude → Qwen after msg 20)" if ACTIVE_PROVIDER == "auto" else f"forced: {ACTIVE_PROVIDER}"
    print(f"Routing:         {routing}")
    print(f"Claude model:    {CLAUDE_MODEL}")
    print(f"OpenRouter model:{OPENROUTER_MODEL}")
    print(f"Switch at msg:   {CLAUDE_MESSAGE_LIMIT}")
    print()

    # Provider status
    print("Providers:")
    print(f"  Claude      {'✓ key set' if ANTHROPIC_KEY else '✗ no ANTHROPIC_API_KEY'}")
    print(f"  OpenRouter  {'✓ key set  model: ' + OPENROUTER_MODEL if OPENROUTER_KEY else '✗ no OPENROUTER_API_KEY'}")
    if "ollama.com" in OLLAMA_URL:
        print(f"  Ollama      ✗ wrong URL — set OLLAMA_BASE_URL=http://localhost:11434")
    else:
        print(f"  Ollama      {OLLAMA_MODEL} → {OLLAMA_URL}")
    print(f"  OpenAI      {'✓ key set' if OPENAI_KEY else '✗ no key'}")
    print()
    print(f"Voice:  {'ElevenLabs ✓' if ELEVENLABS_KEY else 'text only (set ELEVENLABS_API_KEY for audio)'}")
    print()
    print("Type 'quit' to exit. Type 'status' to see current routing.\n")

    while True:
        try:
            user_input = input("[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "status":
            print(f"  Messages sent: {message_count}")
            print(f"  Switch threshold: {CLAUDE_MESSAGE_LIMIT}")
            if message_count >= CLAUDE_MESSAGE_LIMIT:
                print(f"  Current model: OpenRouter/{OPENROUTER_MODEL} (cost mode)")
            else:
                print(f"  Current model: Claude/{CLAUDE_MODEL} (quality mode)")
                print(f"  Switches to Qwen in {CLAUDE_MESSAGE_LIMIT - message_count} more messages")
            continue

        print("[Thinking...]", end="", flush=True)

        try:
            reply, provider, reason = await get_response(user_input)
        except Exception as e:
            print(f"\n[Error] {e}")
            continue

        # Update history and counter
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        message_count += 1

        # Show which model responded and why
        print(f"\r[Alex] ({provider} — {reason}):")
        print(f"{reply}\n")

        if ELEVENLABS_KEY:
            await speak(reply)

        # Warn when approaching the model switch
        if ACTIVE_PROVIDER == "auto" and message_count == CLAUDE_MESSAGE_LIMIT - 2:
            print(f"  [info] Switching to Qwen in 2 messages (msg {CLAUDE_MESSAGE_LIMIT})\n")
        elif ACTIVE_PROVIDER == "auto" and message_count == CLAUDE_MESSAGE_LIMIT:
            print(f"  [info] Now using OpenRouter/{OPENROUTER_MODEL} for cost saving\n")


if __name__ == "__main__":
    asyncio.run(main())
