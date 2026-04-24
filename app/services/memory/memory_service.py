from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.memory import Memory
from app.services.ai.claude_service import extract_memories
from app.core.config import settings
from datetime import datetime


async def get_memories_for_context(
    db: AsyncSession,
    user_id: UUID,
    contact_id: UUID,
    limit: int = None,
) -> str:
    """
    Retrieve memories for a contact — includes both contact-specific memories
    and shared memories (contact_id=None) that all contacts can see.
    """
    limit = limit or settings.MAX_MEMORIES_PER_CONTEXT

    result = await db.execute(
        select(Memory)
        .where(Memory.user_id == user_id)
        .where(
            (Memory.contact_id == contact_id) | (Memory.contact_id == None)
        )
        .order_by(Memory.last_accessed_at.desc().nullslast(), Memory.created_at.desc())
        .limit(limit)
    )
    memories = result.scalars().all()

    if not memories:
        return ""

    for m in memories:
        m.last_accessed_at = datetime.utcnow()
    await db.commit()

    return "\n".join(f"- {m.content}" for m in memories)


async def save_memories_from_conversation(
    db: AsyncSession,
    user_id: UUID,
    contact_id: UUID,
    user_message: str,
    agent_response: str,
    source_message_id: UUID,
) -> None:
    """
    Extract and save new memories after each exchange.

    Facts classified as "shared" are saved with contact_id=None so every
    contact can access them (cross-chat memory). Facts classified as "contact"
    are saved against this specific contact only.

    Shared facts are deduplicated — if the same fact already exists for this
    user we skip it rather than saving duplicates.
    """
    conversation_text = f"User: {user_message}\nAssistant: {agent_response}"
    classified = await extract_memories(conversation_text)

    if not classified:
        return

    # Fetch existing shared fact contents for deduplication (case-insensitive)
    existing_result = await db.execute(
        select(func.lower(Memory.content))
        .where(Memory.user_id == user_id)
        .where(Memory.contact_id == None)
    )
    existing_shared = {row[0] for row in existing_result.all()}

    added = False
    for item in classified:
        fact = item["fact"]
        scope = item.get("scope", "shared")

        if scope == "shared":
            if fact.lower() in existing_shared:
                continue  # already known across all contacts
            memory = Memory(
                user_id=user_id,
                contact_id=None,
                content=fact,
                source_message_id=source_message_id,
            )
            existing_shared.add(fact.lower())
        else:
            memory = Memory(
                user_id=user_id,
                contact_id=contact_id,
                content=fact,
                source_message_id=source_message_id,
            )

        db.add(memory)
        added = True

    if added:
        await db.commit()
