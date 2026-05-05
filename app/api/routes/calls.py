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
import aiohttp
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
from livekit.api.room_service import RoomService, CreateRoomRequest
from uuid import UUID, uuid4
from datetime import datetime

router = APIRouter(prefix="/calls", tags=["calls"])


class StartCallResponse(BaseModel):
    room_name: str
    token: str           # LiveKit JWT for the client
    livekit_url: str     # WebSocket URL for the client to connect to


class StartVideoCallResponse(BaseModel):
    conversation_id: str
    conversation_url: str  # Daily.co URL — open in WebView


class TranscriptPayload(BaseModel):
    user_id: str
    contact_id: str
    room_name: str
    transcript: str
    turn_count: int
    duration_seconds: int


class CallLogEntry(BaseModel):
    id: str                    # message id
    contact_id: str
    contact_name: str
    contact_avatar_emoji: str | None
    contact_avatar_url: str | None
    room_name: str
    turn_count: int
    duration_seconds: int
    called_at: str             # ISO 8601


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


@router.post("/video/{contact_id}", response_model=StartVideoCallResponse)
async def start_video_call(
    contact_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Start a video call with an AI contact via Tavus CVI.
    Returns a Tavus conversation URL to open in a WebView.
    """
    if not settings.TAVUS_API_KEY:
        raise HTTPException(status_code=503, detail="Video calls not configured (missing TAVUS_API_KEY)")
    if not settings.TAVUS_REPLICA_ID:
        raise HTTPException(status_code=503, detail="Video calls not configured (missing TAVUS_REPLICA_ID)")

    result = await db.execute(select(Contact).where(Contact.id == contact_id))
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    memory_context = await get_memories_for_context(db, current_user.id, contact.id)

    # Build the conversational context for Tavus
    user_name = current_user.display_name or "the user"
    system_context = contact.persona_prompt
    system_context += f"\n\nThe person you are speaking with is called {user_name}. Use their name naturally — not in every sentence, just the way a real person would."
    if memory_context:
        system_context += (
            f"\n\n--- Background context ---\n"
            f"{memory_context}\n"
            f"---\n"
            f"This is a fresh call. Start the conversation naturally without referencing "
            f"any of the above. Only bring up past information if the user raises it first "
            f"or it becomes directly relevant to what they are asking."
        )
    system_context += """

You are on a live video call right now. The person can see and hear you, so speak exactly the way a real person does face-to-face on a video call.

Natural call behaviour:
- Match your energy to theirs. Calm if they're calm, engaged if they're animated.
- Respond to what was actually said — don't pre-empt or over-explain.
- Use natural spoken language: contractions, short sentences, the occasional "right" or "sure" — but don't overdo filler.
- Never read out a list. If you need to cover a few things, weave them into natural sentences.
- One point at a time. Say your piece, then let them respond. Don't monologue.
- No "Certainly!", "Absolutely!", "Great question!" — they sound robotic. Just respond directly.
- Keep replies tight: 1–3 sentences unless they're clearly asking for depth.
- End your turn naturally — a soft pause, a gentle question, or simply finishing your thought — so the conversation flows."""

    greeting = f"Hey! Good to hear from you."

    replica_id = contact.avatar_id or settings.TAVUS_REPLICA_ID

    async with aiohttp.ClientSession() as session:
        resp = await session.post(
            "https://tavusapi.com/v2/conversations",
            headers={
                "x-api-key": settings.TAVUS_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "replica_id": replica_id,
                "conversation_name": f"{contact.name} — {current_user.display_name}",
                "conversational_context": system_context,
                "custom_greeting": greeting,
                "properties": {
                    "max_call_duration": 3600,
                    "participant_left_timeout": 60,
                    "enable_recording": False,
                },
            },
        )
        if resp.status not in (200, 201):
            body = await resp.text()
            raise HTTPException(status_code=502, detail=f"Tavus error: {body}")
        data = await resp.json()

    return StartVideoCallResponse(
        conversation_id=data["conversation_id"],
        conversation_url=data["conversation_url"],
    )


@router.post("/video/{conversation_id}/end")
async def end_video_call(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
):
    """End a Tavus conversation when the user hangs up."""
    if not settings.TAVUS_API_KEY:
        return {"status": "ok"}
    async with aiohttp.ClientSession() as session:
        await session.delete(
            f"https://tavusapi.com/v2/conversations/{conversation_id}",
            headers={"x-api-key": settings.TAVUS_API_KEY},
        )
    return {"status": "ended"}


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


@router.get("/history", response_model=list[CallLogEntry])
async def get_call_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns all past call transcripts for the current user, newest first.
    Each entry includes the contact details and call metadata.
    """
    # Fetch all call-transcript messages from chats owned by this user
    result = await db.execute(
        select(Message, Chat)
        .join(Chat, Message.chat_id == Chat.id)
        .where(
            Chat.owner_id == current_user.id,
            Message.content_type == ContentType.voice,
            Message.sender_type == SenderType.agent,
        )
        .order_by(Message.created_at.desc())
    )
    rows = result.all()

    # Filter to only call transcripts (voice notes are also content_type=voice)
    call_rows = [
        (msg, chat) for msg, chat in rows
        if isinstance(msg.meta, dict) and msg.meta.get("type") == "call_transcript"
    ]

    if not call_rows:
        return []

    # Collect unique contact IDs so we can fetch them in one query
    contact_ids = {UUID(msg.sender_id) if isinstance(msg.sender_id, str) else msg.sender_id
                   for msg, _ in call_rows}
    contacts_result = await db.execute(
        select(Contact).where(Contact.id.in_(contact_ids))
    )
    contacts = {c.id: c for c in contacts_result.scalars().all()}

    entries: list[CallLogEntry] = []
    for msg, chat in call_rows:
        contact_id = UUID(msg.sender_id) if isinstance(msg.sender_id, str) else msg.sender_id
        contact = contacts.get(contact_id)
        meta = msg.meta or {}
        entries.append(CallLogEntry(
            id=str(msg.id),
            contact_id=str(contact_id),
            contact_name=contact.name if contact else "Unknown",
            contact_avatar_emoji=contact.avatar_emoji if contact else None,
            contact_avatar_url=contact.avatar_url if contact else None,
            room_name=meta.get("room_name", ""),
            turn_count=meta.get("turn_count", 0),
            duration_seconds=meta.get("duration_seconds", 0),
            called_at=msg.created_at.isoformat(),
        ))

    return entries


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

    # Room metadata — read by the LiveKit agent when it joins via ctx.room.metadata
    room_metadata = json.dumps({
        "contact_id": str(contact.id),
        "user_id": str(current_user.id),
        "user_name": current_user.display_name or "",
        "contact_name": contact.name,
        "persona_prompt": contact.persona_prompt,
        "voice_id": contact.voice_id or "EXAVITQu4vr4xnSDxMaL",
        "memory_context": memory_context,
        "specialty_tags": contact.specialty_tags or [],
    })

    # Create the room with metadata so the agent can read ctx.room.metadata
    livekit_ws = settings.LIVEKIT_URL.replace("wss://", "https://").replace("ws://", "http://")
    async with aiohttp.ClientSession() as http_session:
        room_svc = RoomService(http_session, livekit_ws, settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
        await room_svc.create_room(CreateRoomRequest(
            name=room_name,
            empty_timeout=300,
            metadata=room_metadata,
        ))

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
    for item in memories:
        scope_contact_id = None if item.get("scope") == "shared" else contact_id
        db.add(Memory(
            user_id=user_id,
            contact_id=scope_contact_id,
            content=item["fact"],
            source_message_id=message_id,
        ))
    if memories:
        await db.commit()
