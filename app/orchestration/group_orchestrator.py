"""
Group chat orchestrator.

For every user message in a group chat:
  1. Load all AI contacts from participants
  2. Broadcast typing indicators for all contacts at once
  3. Fire all LLM calls concurrently (parallel, not sequential)
  4. Deliver responses one-by-one, staggered by AGENT_RESPONSE_STAGGER_MS
  5. Save per-contact memories after all messages are delivered

Runs as a FastAPI BackgroundTask so the HTTP response returns immediately
and all agent replies arrive via WebSocket.

Session note: creates its own AsyncSession — the request-scoped session is
already closed by the time this background task executes.
"""
import asyncio
from uuid import UUID
from datetime import datetime

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models.chat import Chat
from app.models.contact import Contact
from app.models.message import Message, SenderType, ContentType
from app.services.ai.claude_service import get_response
from app.services.memory.memory_service import (
    get_memories_for_context,
    save_memories_from_conversation,
)
from app.core.websocket_manager import ws_manager
from app.core.config import settings
from app.orchestration.coordinator import decide_responders


async def handle_group_message(
    chat_id: UUID,
    user_id: UUID,
    user_text: str,
    user_msg_id: UUID,
    history: list[dict],
) -> None:
    """Entry point called as a background task from the messages route."""
    async with AsyncSessionLocal() as db:
        chat_result = await db.execute(select(Chat).where(Chat.id == chat_id))
        chat = chat_result.scalar_one_or_none()
        if not chat:
            return

        contact_ids = [
            UUID(p["id"]) for p in chat.participants if p.get("type") == "contact"
        ]
        if not contact_ids:
            return

        result = await db.execute(
            select(Contact).where(Contact.id.in_(contact_ids))
        )
        all_contacts = {c.id: c for c in result.scalars().all()}
        all_contacts_ordered = [all_contacts[cid] for cid in contact_ids if cid in all_contacts]

        # 1. Coordinator decides which contacts respond and in what order.
        #    Build a brief history summary (last 3 turns) as context.
        history_summary = _summarise_history(history)
        contacts = await decide_responders(user_text, all_contacts_ordered, history_summary)

        # 2. Typing indicators only for selected contacts
        for contact in contacts:
            await ws_manager.send_to_user(str(user_id), {
                "type": "typing",
                "payload": {
                    "chat_id": str(chat.id),
                    "contact_id": str(contact.id),
                    "contact_name": contact.name,
                },
            })

        # 3. Fetch memory contexts for selected contacts concurrently
        memory_contexts = await asyncio.gather(*[
            get_memories_for_context(db, user_id, contact.id)
            for contact in contacts
        ])

        # 4. Fire LLM calls concurrently for selected contacts.
        # force_cheap=True: N contacts × Claude cost is expensive.
        # Qwen via OpenRouter handles group replies at ~20x lower cost.
        responses = await asyncio.gather(*[
            get_response(
                persona_prompt=contact.persona_prompt,
                conversation_history=history,
                user_message=user_text,
                memory_context=memory_ctx,
                force_cheap=True,
            )
            for contact, memory_ctx in zip(contacts, memory_contexts)
        ], return_exceptions=True)

        # 5. Deliver staggered
        stagger_s = settings.AGENT_RESPONSE_STAGGER_MS / 1000
        delivered: list[tuple] = []

        for i, (contact, response) in enumerate(zip(contacts, responses)):
            if isinstance(response, Exception):
                print(f"[Group] {contact.name} failed: {response}")
                continue

            if i > 0:
                await asyncio.sleep(stagger_s)

            agent_msg = Message(
                chat_id=chat.id,
                sender_type=SenderType.agent,
                sender_id=contact.id,
                content_type=ContentType.text,
                text_content=response,
                metadata={"contact_name": contact.name},
            )
            db.add(agent_msg)
            chat.last_message_at = datetime.utcnow()
            await db.commit()
            await db.refresh(agent_msg)

            await ws_manager.send_to_user(str(user_id), {
                "type": "message",
                "payload": {
                    "id": str(agent_msg.id),
                    "chat_id": str(agent_msg.chat_id),
                    "sender_type": agent_msg.sender_type,
                    "sender_id": str(agent_msg.sender_id),
                    "content_type": agent_msg.content_type,
                    "text_content": agent_msg.text_content,
                    "media_url": None,
                    "created_at": agent_msg.created_at.isoformat(),
                    "meta": {"contact_name": contact.name},
                },
            })

            delivered.append((contact, response, agent_msg.id))

        # 6. Save memories after all delivered
        for contact, response, msg_id in delivered:
            try:
                await save_memories_from_conversation(
                    db, user_id, contact.id,
                    user_text, response, msg_id,
                )
            except Exception as e:
                print(f"[Group] Memory save for {contact.name} failed: {e}")


def _summarise_history(history: list[dict], turns: int = 3) -> str:
    """
    Return the last `turns` exchanges as a plain-text string for the
    coordinator's context window. Keeps the prompt short and cheap.
    """
    recent = history[-(turns * 2):]
    lines = []
    for msg in recent:
        prefix = "User" if msg["role"] == "user" else "Agent"
        lines.append(f"{prefix}: {msg['content'][:120]}")
    return "\n".join(lines)


def build_group_history(messages: list) -> list[dict]:
    """
    Build conversation history for the LLM from a list of Message ORM objects.

    Claude's API requires alternating user/assistant turns. In a group chat
    multiple agents reply consecutively, so we merge them into one assistant
    turn, each prefixed with the contact name so every agent sees who said what.

    Example:
      [
        {"role": "user",      "content": "What should I eat?"},
        {"role": "assistant", "content": "[Alex]: Try a salad.\n\n[Maya]: Eat more greens."},
        {"role": "user",      "content": "What about drinks?"},
      ]
    """
    history: list[dict] = []

    for m in messages:
        text = m.text_content or m.transcription or ""
        if not text:
            continue

        if m.sender_type == SenderType.user:
            role = "user"
            content = text
        else:
            role = "assistant"
            name = (m.meta or {}).get("contact_name", "Agent")
            content = f"[{name}]: {text}"

        if history and history[-1]["role"] == role:
            history[-1]["content"] += f"\n\n{content}"
        else:
            history.append({"role": role, "content": content})

    return history
