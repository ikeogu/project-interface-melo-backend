from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.chat import Chat, ChatType
from app.models.contact import Contact
from app.schemas.chat import ChatCreate, GroupChatCreate, ChatOut
from uuid import UUID

router = APIRouter(prefix="/chats", tags=["chats"])


@router.get("/", response_model=list[ChatOut])
async def list_chats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Chat).where(Chat.owner_id == current_user.id).order_by(Chat.last_message_at.desc().nullslast())
    )
    return result.scalars().all()


@router.post("/direct", response_model=ChatOut)
async def create_direct_chat(
    payload: ChatCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    contact_result = await db.execute(select(Contact).where(Contact.id == payload.contact_id))
    contact = contact_result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    chat = Chat(
        owner_id=current_user.id,
        chat_type=ChatType.direct,
        name=contact.name,
        participants=[
            {"id": str(current_user.id), "type": "user"},
            {"id": str(contact.id), "type": "contact"},
        ],
    )
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return chat


@router.post("/group", response_model=ChatOut)
async def create_group_chat(
    payload: GroupChatCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    participants = [{"id": str(current_user.id), "type": "user"}]
    for cid in payload.contact_ids:
        result = await db.execute(select(Contact).where(Contact.id == cid))
        contact = result.scalar_one_or_none()
        if contact:
            participants.append({"id": str(cid), "type": "contact"})

    chat = Chat(
        owner_id=current_user.id,
        chat_type=ChatType.group,
        name=payload.name,
        participants=participants,
    )
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return chat
