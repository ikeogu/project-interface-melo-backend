"""
LLM service — three-tier routing:

  Tier 1 — Claude (Anthropic)       messages 1-N    best quality
  Tier 2 — Qwen   (OpenRouter)      messages N+     cost saving
  Tier 3 — Ollama (self-hosted)     dev fallback    free, local

Tier 3 only activates when Tiers 1 & 2 both fail or have no key set.
In production you will never hit Ollama — it exists so dev works
without spending credits.

Ollama uses the OpenAI-compatible endpoint (/v1/chat/completions)
so the code path is identical to OpenRouter.

Config (.env):
  ANTHROPIC_API_KEY=...
  OPENROUTER_API_KEY=...
  OLLAMA_BASE_URL=http://localhost:11434   (default)
  OLLAMA_MODEL=llama3.2                    (default)
  CLAUDE_MESSAGE_LIMIT=20                  (default)
"""
import anthropic
import httpx
import json
from typing import AsyncGenerator
from app.core.config import settings

claude_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

OPENROUTER_BASE  = "https://openrouter.ai/api/v1"
QWEN_MODEL       = "qwen/qwen3-235b-a22b"
CLAUDE_MESSAGE_LIMIT = 20


# ── Routing decision ───────────────────────────────────────────────────────────

def _pick_tier(message_count: int, force_cheap: bool = False) -> str:
    """
    Returns which tier to try first: 'claude' | 'qwen' | 'ollama'.
    Ollama is never the first choice — it's always a fallback.
    """
    if force_cheap or message_count >= CLAUDE_MESSAGE_LIMIT:
        return "qwen"
    return "claude"


# ── Public API ─────────────────────────────────────────────────────────────────

async def stream_response(
    persona_prompt: str,
    conversation_history: list[dict],
    user_message: str,
    memory_context: str = "",
    message_count: int = 0,
    force_cheap: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Stream response token by token.
    Tries primary tier first, falls back down the chain automatically.
    """
    tier = _pick_tier(message_count, force_cheap)
    system = _build_system(persona_prompt, memory_context)
    messages = conversation_history + [{"role": "user", "content": user_message}]

    if tier == "claude":
        try:
            async for token in _stream_claude(system, messages):
                yield token
            return
        except Exception as e:
            print(f"[LLM] Claude stream failed: {e} — falling back")

    # Qwen fallback
    try:
        async for token in _stream_openrouter(system, messages, QWEN_MODEL):
            yield token
        return
    except Exception as e:
        print(f"[LLM] OpenRouter stream failed: {e} — falling back to Ollama")

    # Ollama dev fallback
    try:
        async for token in _stream_ollama(system, messages):
            yield token
        return
    except Exception as e:
        print(f"[LLM] Ollama stream failed: {e}")

    # Last resort: Claude (even if force_cheap was requested)
    if settings.ANTHROPIC_API_KEY:
        print("[LLM] All cheap options failed — falling back to Claude")
        async for token in _stream_claude(system, messages):
            yield token
        return

    raise Exception("No LLM provider available")


async def get_response(
    persona_prompt: str,
    conversation_history: list[dict],
    user_message: str,
    memory_context: str = "",
    message_count: int = 0,
    force_cheap: bool = False,
) -> str:
    """
    Full (non-streaming) response.
    Same three-tier fallback as stream_response.
    """
    tier = _pick_tier(message_count, force_cheap)
    system = _build_system(persona_prompt, memory_context)
    messages = conversation_history + [{"role": "user", "content": user_message}]

    if tier == "claude":
        try:
            return await _get_claude(system, messages)
        except Exception as e:
            print(f"[LLM] Claude failed: {e} — falling back")

    try:
        return await _get_openrouter(system, messages, QWEN_MODEL)
    except Exception as e:
        print(f"[LLM] OpenRouter failed: {e} — falling back to Ollama")

    try:
        return await _get_ollama(system, messages)
    except Exception as e:
        print(f"[LLM] Ollama failed: {e}")

    # Last resort: Claude (even if force_cheap was requested)
    if settings.ANTHROPIC_API_KEY:
        print("[LLM] All cheap options failed — falling back to Claude")
        return await _get_claude(system, messages)

    raise Exception("No LLM provider available")


# ── Internal tasks (always use cheapest available) ─────────────────────────────

async def get_cheap_completion(system: str, user_message: str) -> str:
    """
    Raw system+message call using the cheapest available model.
    Used for routing/classification tasks (coordinator, etc.) where
    we need full control over the prompt without the persona wrapper.
    """
    messages = [{"role": "user", "content": user_message}]
    try:
        return await _get_openrouter(system, messages, QWEN_MODEL)
    except Exception as e:
        print(f"[LLM] OpenRouter cheap completion failed: {e} — falling back to Ollama")
    return await _get_ollama(system, messages)


async def extract_memories(conversation_text: str) -> list[dict]:
    """
    Extract memorable facts and classify each as:
      "shared"  — about the user themselves (any contact should know this)
      "contact" — about this specific relationship/conversation (only this contact)

    Returns list of {"fact": str, "scope": "shared"|"contact"}.
    """
    system = (
        "Extract key facts from this conversation and classify each.\n"
        "scope=shared : facts about the user themselves — location, job, goals, "
        "life events, general preferences. Any AI contact should know these.\n"
        "scope=contact : facts specific to THIS contact relationship — how the user "
        "likes this contact to respond, topics unique to this thread.\n"
        "Return ONLY a JSON array, or [] if nothing memorable.\n"
        'Example: [{"fact": "User is a Lagos-based founder", "scope": "shared"}, '
        '{"fact": "User prefers bullet points in replies from this contact", "scope": "contact"}]'
    )

    if not settings.OPENROUTER_API_KEY:
        return []

    try:
        text = await _get_openrouter(
            system,
            [{"role": "user", "content": conversation_text}],
            QWEN_MODEL,
        )
    except Exception:
        return []

    try:
        raw = json.loads(text)
        # Accept both old format (list[str]) and new format (list[dict])
        result = []
        for item in raw:
            if isinstance(item, str):
                result.append({"fact": item, "scope": "shared"})
            elif isinstance(item, dict) and "fact" in item:
                result.append({"fact": item["fact"], "scope": item.get("scope", "shared")})
        return result
    except Exception:
        return []


async def generate_persona_prompt(name: str, description: str) -> str:
    """Auto-generate a persona prompt — uses cheapest available model."""
    system = (
        "Generate a persona system prompt for an AI contact in a messaging app. "
        "Given a name and description, write a structured prompt under 200 words "
        "that makes the contact feel like a real person, not a tool."
    )
    user = f"Name: {name}\nDescription: {description}"
    try:
        return await _get_openrouter(system, [{"role": "user", "content": user}], QWEN_MODEL)
    except Exception:
        return await _get_ollama(system, [{"role": "user", "content": user}])


# ── Tier 1 — Claude ────────────────────────────────────────────────────────────

async def _stream_claude(system: str, messages: list[dict]) -> AsyncGenerator[str, None]:
    if not settings.ANTHROPIC_API_KEY:
        raise Exception("ANTHROPIC_API_KEY not set")
    async with claude_client.messages.stream(
        model=settings.CLAUDE_MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def _get_claude(system: str, messages: list[dict]) -> str:
    if not settings.ANTHROPIC_API_KEY:
        raise Exception("ANTHROPIC_API_KEY not set")
    r = await claude_client.messages.create(
        model=settings.CLAUDE_MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return r.content[0].text


# ── Tier 2 — OpenRouter (Qwen / any model) ─────────────────────────────────────

async def _stream_openrouter(
    system: str, messages: list[dict], model: str
) -> AsyncGenerator[str, None]:
    if not settings.OPENROUTER_API_KEY:
        raise Exception("OPENROUTER_API_KEY not set")

    full_messages = [{"role": "system", "content": system}] + messages
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "HTTP-Referer": settings.APP_BASE_URL,
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": full_messages, "max_tokens": 1024, "stream": True},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue


async def _get_openrouter(system: str, messages: list[dict], model: str) -> str:
    if not settings.OPENROUTER_API_KEY:
        raise Exception("OPENROUTER_API_KEY not set")

    full_messages = [{"role": "system", "content": system}] + messages
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "HTTP-Referer": settings.APP_BASE_URL,
            },
            json={"model": model, "messages": full_messages, "max_tokens": 1024},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ── Tier 3 — Ollama (dev fallback, OpenAI-compatible endpoint) ─────────────────

async def _stream_ollama(system: str, messages: list[dict]) -> AsyncGenerator[str, None]:
    """
    Ollama exposes /v1/chat/completions — identical to OpenAI API.
    Works in dev without any API keys or credits.
    Never reached in production (Claude and OpenRouter both up).
    """
    _assert_ollama_url()
    full_messages = [{"role": "system", "content": system}] + messages

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{settings.OLLAMA_BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": full_messages,
                "max_tokens": 1024,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue


async def _get_ollama(system: str, messages: list[dict]) -> str:
    _assert_ollama_url()
    full_messages = [{"role": "system", "content": system}] + messages

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{settings.OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": full_messages,
                "max_tokens": 1024,
                "stream": False,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _assert_ollama_url():
    if not settings.OLLAMA_BASE_URL or "ollama.com" in settings.OLLAMA_BASE_URL:
        raise Exception(
            "Ollama not configured. Set OLLAMA_BASE_URL=http://localhost:11434 "
            "and run: ollama serve && ollama pull llama3.2"
        )


# ── System prompt builder ──────────────────────────────────────────────────────

def _build_system(persona_prompt: str, memory_context: str) -> str:
    if not memory_context:
        return persona_prompt
    return (
        f"{persona_prompt}\n\n"
        f"--- What you know about this user ---\n"
        f"{memory_context}\n"
        f"---"
    )
