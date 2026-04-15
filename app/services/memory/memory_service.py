from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
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
    Retrieve relevant memories for a contact-user pair.
    Returns formatted string ready to inject into the system prompt.
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

    # Update last_accessed_at for retrieved memories
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
    Async job: extract and save new memories after each exchange.
    Runs after the response is already delivered to the user.
    """
    conversation_text = f"User: {user_message}\nAssistant: {agent_response}"
    new_facts = await extract_memories(conversation_text)

    for fact in new_facts:
        memory = Memory(
            user_id=user_id,
            contact_id=contact_id,
            content=fact,
            source_message_id=source_message_id,
        )
        db.add(memory)

    if new_facts:
        await db.commit()
