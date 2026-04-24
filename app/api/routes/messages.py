from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.chat import Chat, ChatType
from app.models.contact import Contact
from app.models.message import Message, SenderType, ContentType
from app.schemas.message import MessageCreate, MessageOut
from app.services.ai.claude_service import get_response
from app.services.memory.memory_service import get_memories_for_context, save_memories_from_conversation
from app.services.voice.stt_service import transcribe_audio
from app.services.voice.tts_service import synthesise_speech
from app.services.voice.storage_service import upload_audio
from app.core.websocket_manager import ws_manager
from app.orchestration.group_orchestrator import handle_group_message, build_group_history
from datetime import datetime
from uuid import UUID

router = APIRouter(prefix="/messages", tags=["messages"])


async def _get_conversation_history(db: AsyncSession, chat_id: UUID, limit: int = 20) -> list[dict]:
    result = await db.execute(
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    msgs = list(reversed(result.scalars().all()))
    return [
        {
            "role": "user" if m.sender_type == SenderType.user else "assistant",
            "content": m.text_content or m.transcription or "",
        }
        for m in msgs if (m.text_content or m.transcription)
    ]


async def _get_conversation_messages(db: AsyncSession, chat_id: UUID, limit: int = 20) -> list:
    """Return raw Message objects for history builders that need sender metadata."""
    result = await db.execute(
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    return list(reversed(result.scalars().all()))


@router.get("/{chat_id}", response_model=list[MessageOut])
async def get_messages(
    chat_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at)
    )
    return result.scalars().all()


@router.post("/", response_model=MessageOut)
async def send_message(
    payload: MessageCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # 1. Verify chat belongs to user
    chat_result = await db.execute(
        select(Chat).where(Chat.id == payload.chat_id, Chat.owner_id == current_user.id)
    )
    chat = chat_result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # 2. Transcribe voice note if needed
    transcription = None
    user_text = payload.text_content

    if payload.content_type == ContentType.voice and payload.media_url:
        transcription = await transcribe_audio(payload.media_url)
        user_text = transcription

    # 3. Save user message
    user_msg = Message(
        chat_id=payload.chat_id,
        sender_type=SenderType.user,
        sender_id=current_user.id,
        content_type=payload.content_type,
        text_content=payload.text_content,
        media_url=payload.media_url,
        transcription=transcription,
    )
    db.add(user_msg)
    await db.commit()
    await db.refresh(user_msg)

    # 4a. GROUP CHAT — delegate to orchestrator as background task and return immediately.
    #     All agent replies arrive via WebSocket; no HTTP timeout risk.
    if chat.chat_type == ChatType.group:
        raw_msgs = await _get_conversation_messages(db, payload.chat_id)
        history = build_group_history(raw_msgs)
        background_tasks.add_task(
            handle_group_message,
            chat.id,
            current_user.id,
            user_text or "",
            user_msg.id,
            history,
        )
        return MessageOut.model_validate(user_msg)

    # 4b. DIRECT CHAT — single contact, synchronous response
    contact_participant = next((p for p in chat.participants if p.get("type") == "contact"), None)
    if not contact_participant:
        raise HTTPException(status_code=400, detail="No AI contact in this chat")

    contact_result = await db.execute(select(Contact).where(Contact.id == contact_participant["id"]))
    contact = contact_result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    memory_context = await get_memories_for_context(db, current_user.id, contact.id)
    history = await _get_conversation_history(db, payload.chat_id)

    response_text = await get_response(
        persona_prompt=contact.persona_prompt,
        conversation_history=history,
        user_message=user_text,
        memory_context=memory_context,
    )

    # TTS for voice notes
    agent_media_url = None
    agent_content_type = ContentType.text

    if payload.content_type == ContentType.voice:
        audio_bytes = await synthesise_speech(response_text, voice_id=contact.voice_id)
        agent_media_url = await upload_audio(audio_bytes, content_type="audio/mpeg")
        agent_content_type = ContentType.voice

    agent_msg = Message(
        chat_id=payload.chat_id,
        sender_type=SenderType.agent,
        sender_id=contact.id,
        content_type=agent_content_type,
        text_content=response_text,
        media_url=agent_media_url,
        metadata={"contact_name": contact.name},
    )
    db.add(agent_msg)
    chat.last_message_at = datetime.utcnow()
    await db.commit()
    await db.refresh(agent_msg)

    await ws_manager.send_to_user(str(current_user.id), {
        "type": "message",
        "payload": {
            "id": str(agent_msg.id),
            "chat_id": str(agent_msg.chat_id),
            "sender_type": agent_msg.sender_type,
            "sender_id": str(agent_msg.sender_id),
            "content_type": agent_msg.content_type,
            "text_content": agent_msg.text_content,
            "media_url": agent_msg.media_url,
            "created_at": agent_msg.created_at.isoformat(),
        },
    })

    background_tasks.add_task(
        save_memories_from_conversation,
        db, current_user.id, contact.id,
        user_text, response_text, user_msg.id,
    )

    return MessageOut.model_validate(user_msg)
