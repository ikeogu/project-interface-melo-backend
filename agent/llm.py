"""
LLM module for the call agent.

Same Claude → Qwen routing as the main API but optimised for
real-time conversation:
  - Shorter max_tokens (we're speaking, not writing essays)
  - Streaming enabled so TTS can start before full response is ready
  - System prompt includes "you are on a live voice call" instruction
    so the model keeps responses conversational and concise
"""
import anthropic
import httpx
import json
from typing import AsyncGenerator
from dataclasses import dataclass

CLAUDE_MODEL = "claude-sonnet-4-20250514"
QWEN_MODEL = "qwen/qwen3-235b-a22b"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Voice call responses should be short — no markdown, no lists
VOICE_SYSTEM_SUFFIX = """

IMPORTANT — you are on a live voice call right now.
Rules for voice responses:
- Keep answers under 3 sentences unless the user explicitly asks for more
- Never use bullet points, markdown, headers, or lists
- Speak naturally, like a person on a phone call
- Don't say "Certainly!" or "Of course!" — just answer directly
- If you need to think, say "Give me a moment" naturally
"""

CLAUDE_TURN_LIMIT = 15  # switch to Qwen after this many turns


@dataclass
class LLMConfig:
    anthropic_key: str
    openrouter_key: str
    app_base_url: str = "http://localhost:8000"


class CallLLM:
    """LLM handler for the real-time call pipeline."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._claude = anthropic.AsyncAnthropic(api_key=config.anthropic_key)

    async def stream_response(
        self,
        persona_prompt: str,
        history: list[dict],
        user_message: str,
        memory_context: str = "",
        turn_count: int = 0,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the LLM response token by token.
        Caller can pipe this directly into the TTS engine.
        """
        system = _build_voice_system(persona_prompt, memory_context)
        messages = history + [{"role": "user", "content": user_message}]

        use_qwen = turn_count > CLAUDE_TURN_LIMIT
        provider = "qwen" if use_qwen else "claude"
        print(f"[LLM] Turn {turn_count} → using {provider}")

        if use_qwen:
            async for token in self._stream_qwen(system, messages):
                yield token
        else:
            async for token in self._stream_claude(system, messages):
                yield token

    async def _stream_claude(
        self, system: str, messages: list[dict]
    ) -> AsyncGenerator[str, None]:
        async with self._claude.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=300,        # short for voice
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def _stream_qwen(
        self, system: str, messages: list[dict]
    ) -> AsyncGenerator[str, None]:
        full_messages = [{"role": "system", "content": system}] + messages
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream(
                "POST",
                f"{OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.openrouter_key}",
                    "HTTP-Referer": self.config.app_base_url,
                    "Content-Type": "application/json",
                },
                json={
                    "model": QWEN_MODEL,
                    "messages": full_messages,
                    "max_tokens": 300,
                    "stream": True,
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                        except Exception:
                            continue


def _build_voice_system(persona_prompt: str, memory_context: str) -> str:
    """Build system prompt with voice-specific instructions."""
    base = persona_prompt + VOICE_SYSTEM_SUFFIX
    if memory_context:
        base += f"\n\n--- What you know about this user ---\n{memory_context}\n---"
    return base
