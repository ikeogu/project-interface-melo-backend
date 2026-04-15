"""
Calls routes.

When a user taps "Call" on a contact:
  1. Backend creates a LiveKit room with contact metadata in room.metadata
  2. Backend generates a LiveKit access token for the user
  3. Backend signals the LiveKit Agent to join the room
  4. Frontend connects to the room using the token
  5. Agent joins automatically via LiveKit Workers (listening for new rooms)

POST /calls/transcript is called by the agent after the call ends
to save the transcript and trigger memory extraction.
"""
import json
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.contact import Contact
from app.models.chat import Chat, ChatType
from app.models.message import Message, SenderType, ContentType
from app.core.config import settings
from app.services.memory.memory_service import (
    get_memories_for_context,
    save_memories_from_conversation,
)
from livekit.api import AccessToken, VideoGrants
from uuid import UUID, uuid4
from datetime import datetime

router = APIRouter(prefix="/calls", tags=["calls"])


class StartCallResponse(BaseModel):
    room_name: str
    token: str           # LiveKit JWT for the client
    livekit_url: str     # WebSocket URL for the client to connect to


class TranscriptPayload(BaseModel):
    user_id: str
    contact_id: str
    room_name: str
    transcript: str
    turn_count: int
    duration_seconds: int


@router.post("/voice/{contact_id}", response_model=StartCallResponse)
async def start_voice_call(
    contact_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Start an audio call with an AI contact.
    Returns a LiveKit token for the client.
    The agent joins automatically via LiveKit Workers.
    """
    return await _create_call_session(contact_id, current_user, db, video=False)


@router.post("/video/{contact_id}", response_model=StartCallResponse)
async def start_video_call(
    contact_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Start a video call with an AI contact.
    Uses Tavus for the avatar (Phase 3). For now returns same as voice.
    """
    return await _create_call_session(contact_id, current_user, db, video=True)


@router.post("/transcript")
async def save_transcript(
    payload: TranscriptPayload,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Called by the LiveKit agent after a call ends.
    Saves transcript as messages and queues memory extraction.
    """
    # Find or create a direct chat between user and contact
    chat = await _get_or_create_chat(
        db,
        user_id=UUID(payload.user_id),
        contact_id=UUID(payload.contact_id),
    )

    # Save transcript as a single voice/call message
    msg = Message(
        chat_id=chat.id,
        sender_type=SenderType.agent,
        sender_id=UUID(payload.contact_id),
        content_type=ContentType.voice,
        text_content=payload.transcript,
        metadata={
            "type": "call_transcript",
            "turn_count": payload.turn_count,
            "duration_seconds": payload.duration_seconds,
            "room_name": payload.room_name,
        },
    )
    db.add(msg)
    chat.last_message_at = datetime.utcnow()
    await db.commit()
    await db.refresh(msg)

    # Extract memories from transcript in background
    background_tasks.add_task(
        _extract_call_memories,
        db=db,
        user_id=UUID(payload.user_id),
        contact_id=UUID(payload.contact_id),
        transcript=payload.transcript,
        message_id=msg.id,
    )

    return {"status": "saved", "message_id": str(msg.id)}


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _create_call_session(
    contact_id: UUID,
    current_user: User,
    db: AsyncSession,
    video: bool = False,
) -> StartCallResponse:
    """Create a LiveKit room and token for a call."""
    result = await db.execute(select(Contact).where(Contact.id == contact_id))
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    # Load memory context for this contact
    memory_context = await get_memories_for_context(db, current_user.id, contact.id)

    # Create a unique room name
    room_name = f"call-{current_user.id}-{contact.id}-{uuid4().hex[:8]}"

    # Room metadata — read by the LiveKit agent when it joins
    room_metadata = json.dumps({
        "contact_id": str(contact.id),
        "user_id": str(current_user.id),
        "contact_name": contact.name,
        "persona_prompt": contact.persona_prompt,
        "voice_id": contact.voice_id or "af_heart",
        "memory_context": memory_context,
        "specialty_tags": contact.specialty_tags or [],
    })

    # Generate LiveKit access token for the user
    token = (
        AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
        .with_identity(str(current_user.id))
        .with_name(current_user.display_name)
        .with_grants(VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        ))
        .with_metadata(room_metadata)
        .to_jwt()
    )

    return StartCallResponse(
        room_name=room_name,
        token=token,
        livekit_url=settings.LIVEKIT_URL,
    )


async def _get_or_create_chat(db: AsyncSession, user_id: UUID, contact_id: UUID) -> Chat:
    """Find existing direct chat or create one."""
    result = await db.execute(
        select(Chat).where(
            Chat.owner_id == user_id,
            Chat.chat_type == ChatType.direct,
        )
    )
    chats = result.scalars().all()
    for chat in chats:
        participants = chat.participants or []
        ids = [p.get("id") for p in participants]
        if str(contact_id) in ids:
            return chat

    # Create new chat
    chat = Chat(
        owner_id=user_id,
        chat_type=ChatType.direct,
        participants=[
            {"id": str(user_id), "type": "user"},
            {"id": str(contact_id), "type": "contact"},
        ],
    )
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return chat


async def _extract_call_memories(db, user_id, contact_id, transcript, message_id):
    """Background task: extract memories from call transcript."""
    from app.services.ai.claude_service import extract_memories
    from app.models.memory import Memory

    memories = await extract_memories(transcript)
    for fact in memories:
        db.add(Memory(
            user_id=user_id,
            contact_id=contact_id,
            content=fact,
            source_message_id=message_id,
        ))
    if memories:
        await db.commit()
