"""
Coordinator agent — decides which contacts should respond in a group chat
and in what order.

Uses the cheapest available LLM (Qwen via OpenRouter) since this is a
routing/classification decision, not content generation.

Always falls back to all contacts if the call fails or returns garbage,
so group chats never break because of the coordinator.
"""
import json
from app.models.contact import Contact
from app.services.ai.claude_service import get_cheap_completion


async def decide_responders(
    user_text: str,
    contacts: list[Contact],
    history_summary: str = "",
) -> list[Contact]:
    """
    Return an ordered subset of contacts who should respond to user_text.

    - Returns 1–3 contacts for focused messages
    - Returns all contacts when the user addresses everyone
    - Always returns at least 1 contact
    - Preserves order (most relevant first)
    - Falls back to all contacts on any error
    """
    if len(contacts) <= 1:
        return contacts

    contact_lines = "\n".join(
        f"- {c.name}: {', '.join(c.specialty_tags or ['general'])}"
        for c in contacts
    )

    context_section = (
        f"\nRecent conversation:\n{history_summary}\n" if history_summary else ""
    )

    system = (
        "You are a group chat coordinator. Decide which AI contacts should reply "
        "to the user's message and in what order.\n\n"
        "Rules:\n"
        "1. Return 1–3 contacts unless the message clearly addresses everyone.\n"
        "2. Put the most relevant contact first.\n"
        "3. If the message says 'what do you all think', 'everyone', 'all of you', "
        "or similar → return all contacts.\n"
        "4. Return ONLY a valid JSON array of names — nothing else.\n"
        "   Example: [\"Maya\", \"Alex\"]\n"
        "5. Never return an empty array.\n\n"
        f"Available contacts:\n{contact_lines}"
    )

    user_prompt = f"{context_section}User message: {user_text}"

    try:
        raw = await get_cheap_completion(system, user_prompt)

        # Extract JSON array even if the model wraps it in prose
        raw = raw.strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("no JSON array in response")

        names = json.loads(raw[start:end])
        if not isinstance(names, list) or not names:
            raise ValueError("empty or invalid array")

        # Map back to Contact objects (case-insensitive, preserve coordinator order)
        name_map = {c.name.lower(): c for c in contacts}
        ordered = [name_map[n.lower()] for n in names if n.lower() in name_map]

        if not ordered:
            raise ValueError("no recognised contact names returned")

        print(f"[Coordinator] '{user_text[:60]}' → {[c.name for c in ordered]}")
        return ordered

    except Exception as e:
        print(f"[Coordinator] Failed ({e}), falling back to all contacts")
        return contacts
