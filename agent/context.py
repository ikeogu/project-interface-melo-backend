"""
CallContext — everything the agent needs to know about a specific call.

Loaded from the database at call start using the contact_id and user_id
passed in the LiveKit room metadata. Holds persona, memory, and
conversation history for the duration of the call.
"""
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CallContext:
    # Identifiers
    room_name: str
    user_id: str
    contact_id: str
    contact_name: str

    # Persona + memory (loaded from DB at call start)
    persona_prompt: str
    memory_context: str = ""

    # Conversation history built up during the call
    # Each entry: {"role": "user"|"assistant", "content": str, "ts": datetime}
    history: list[dict] = field(default_factory=list)

    # Stats for routing (Claude vs Qwen)
    turn_count: int = 0
    call_started_at: datetime = field(default_factory=datetime.utcnow)

    # State flags
    is_agent_speaking: bool = False
    is_user_speaking: bool = False

    def add_user_turn(self, text: str):
        self.history.append({
            "role": "user",
            "content": text,
            "ts": datetime.utcnow(),
        })
        self.turn_count += 1

    def add_agent_turn(self, text: str):
        self.history.append({
            "role": "assistant",
            "content": text,
            "ts": datetime.utcnow(),
        })

    def get_llm_history(self, max_turns: int = 20) -> list[dict]:
        """Return last N turns in LLM-compatible format (no ts field)."""
        recent = self.history[-max_turns:]
        return [{"role": h["role"], "content": h["content"]} for h in recent]

    def get_full_transcript(self) -> str:
        """Formatted transcript for saving to DB after call ends."""
        lines = []
        for h in self.history:
            speaker = "User" if h["role"] == "user" else self.contact_name
            ts = h["ts"].strftime("%H:%M:%S")
            lines.append(f"[{ts}] {speaker}: {h['content']}")
        return "\n".join(lines)
