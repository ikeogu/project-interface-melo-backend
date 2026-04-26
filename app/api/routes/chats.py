from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.chat import Chat, ChatType
from app.models.contact import Contact
from app.models.user import User as UserModel
from app.schemas.chat import ChatCreate, GroupChatCreate, ChatOut, InviteUser
from uuid import UUID

router = APIRouter(prefix="/chats", tags=["chats"])

# ---------------------------------------------------------------------------
# Pre-built group templates — each references contact templates by name.
# Adding a new template here is the only change needed to expose it in the app.
# ---------------------------------------------------------------------------
_GROUP_TEMPLATES = [
    {
        "key": "ceo_board",
        "name": "CEO Board",
        "emoji": "🏢",
        "description": "Your executive advisory team for big decisions",
        "member_names": ["Alex", "Marcus"],
    },
    {
        "key": "wellness_circle",
        "name": "Wellness Circle",
        "emoji": "🌿",
        "description": "Holistic mind, body, and spirit support",
        "member_names": ["Maya", "Father James"],
    },
    {
        "key": "life_panel",
        "name": "Life Panel",
        "emoji": "✨",
        "description": "Your complete personal advisory board",
        "member_names": ["Alex", "Maya", "Father James"],
    },
    {
        "key": "growth_team",
        "name": "Growth Team",
        "emoji": "🚀",
        "description": "Business strategy meets wellness — scale without burning out",
        "member_names": ["Alex", "Marcus", "Maya"],
    },
]


class GroupTemplateRequest(BaseModel):
    template_key: str


@router.get("/", response_model=list[ChatOut])
async def list_chats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Chat).where(Chat.owner_id == current_user.id).order_by(Chat.last_message_at.desc().nullslast())
    )
    chats = result.scalars().all()

    # Deduplicate direct chats: keep only the most recently active chat per contact.
    # (Duplicates accumulate when the client calls POST /chats/direct multiple times.)
    seen_contact_ids: set[str] = set()
    deduped: list[Chat] = []
    for chat in chats:
        if chat.chat_type != ChatType.direct:
            deduped.append(chat)
            continue
        contact_id = next(
            (p["id"] for p in (chat.participants or []) if p.get("type") == "contact"),
            None,
        )
        if contact_id and contact_id not in seen_contact_ids:
            seen_contact_ids.add(contact_id)
            deduped.append(chat)

    return deduped


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

    # Return existing direct chat if one already exists for this user-contact pair
    existing_result = await db.execute(
        select(Chat).where(
            Chat.owner_id == current_user.id,
            Chat.chat_type == ChatType.direct,
        )
    )
    for chat in existing_result.scalars().all():
        ids = [p.get("id") for p in (chat.participants or [])]
        if str(payload.contact_id) in ids:
            return chat

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
    from app.core.config import settings

    if len(payload.contact_ids) < 2:
        raise HTTPException(status_code=400, detail="A group chat needs at least 2 contacts")

    if len(payload.contact_ids) > settings.GROUP_CHAT_MAX_AGENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.GROUP_CHAT_MAX_AGENTS} contacts per group chat",
        )

    participants = [{"id": str(current_user.id), "type": "user"}]

    for cid in payload.contact_ids:
        result = await db.execute(
            select(Contact).where(
                Contact.id == cid,
                (Contact.owner_id == current_user.id) | (Contact.is_template == True),
            )
        )
        contact = result.scalar_one_or_none()
        if not contact:
            raise HTTPException(status_code=404, detail=f"Contact {cid} not found")
        participants.append({"id": str(cid), "type": "contact"})

    # Optional: invite other real users → makes this a mixed chat
    for uid in payload.user_ids:
        if uid == current_user.id:
            continue
        user_result = await db.execute(select(UserModel).where(UserModel.id == uid))
        invited = user_result.scalar_one_or_none()
        if not invited:
            raise HTTPException(status_code=404, detail=f"User {uid} not found")
        participants.append({"id": str(uid), "type": "user"})

    chat_type = ChatType.mixed if payload.user_ids else ChatType.group

    chat = Chat(
        owner_id=current_user.id,
        chat_type=chat_type,
        name=payload.name,
        participants=participants,
    )
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return chat


@router.post("/{chat_id}/invite", response_model=ChatOut)
async def invite_user_to_chat(
    chat_id: UUID,
    payload: InviteUser,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Add a real user to an existing group or mixed chat by email."""
    chat_result = await db.execute(
        select(Chat).where(Chat.id == chat_id, Chat.owner_id == current_user.id)
    )
    chat = chat_result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.chat_type == ChatType.direct:
        raise HTTPException(status_code=400, detail="Cannot invite users to a direct chat")

    user_result = await db.execute(
        select(UserModel).where(UserModel.email == payload.email)
    )
    invitee = user_result.scalar_one_or_none()
    if not invitee:
        raise HTTPException(status_code=404, detail="User not found")

    already_in = any(p["id"] == str(invitee.id) for p in chat.participants)
    if already_in:
        raise HTTPException(status_code=400, detail="User is already in this chat")

    chat.participants = chat.participants + [{"id": str(invitee.id), "type": "user"}]
    chat.chat_type = ChatType.mixed
    await db.commit()
    await db.refresh(chat)
    return chat


@router.get("/group-templates")
async def list_group_templates(db: AsyncSession = Depends(get_db)):
    """
    Return pre-built group template configs with resolved contact template IDs.
    The frontend uses this to show a 'Start from template' gallery.
    """
    result = await db.execute(select(Contact).where(Contact.is_template == True))
    template_map = {c.name: c for c in result.scalars().all()}

    out = []
    for tpl in _GROUP_TEMPLATES:
        members = []
        available = True
        for name in tpl["member_names"]:
            contact = template_map.get(name)
            if not contact:
                available = False
                break
            members.append({
                "id": str(contact.id),
                "name": contact.name,
                "avatar_emoji": contact.avatar_emoji,
                "specialty_tags": contact.specialty_tags,
            })
        if available:
            out.append({
                "key": tpl["key"],
                "name": tpl["name"],
                "emoji": tpl["emoji"],
                "description": tpl["description"],
                "members": members,
            })
    return out


@router.post("/from-group-template", response_model=ChatOut)
async def create_chat_from_group_template(
    payload: GroupTemplateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    One-tap group chat creation from a pre-built template.
    Creates personal copies of any template contacts the user doesn't have yet,
    then creates the group chat with all of them.
    """
    tpl = next((t for t in _GROUP_TEMPLATES if t["key"] == payload.template_key), None)
    if not tpl:
        raise HTTPException(status_code=404, detail="Group template not found")

    participants = [{"id": str(current_user.id), "type": "user"}]

    for name in tpl["member_names"]:
        tmpl_result = await db.execute(
            select(Contact).where(Contact.name == name, Contact.is_template == True)
        )
        template_contact = tmpl_result.scalar_one_or_none()
        if not template_contact:
            raise HTTPException(
                status_code=404,
                detail=f"Contact template '{name}' not found — run seed script",
            )

        existing_result = await db.execute(
            select(Contact).where(
                Contact.owner_id == current_user.id,
                Contact.name == name,
            )
        )
        personal = existing_result.scalar_one_or_none()

        if not personal:
            personal = Contact(
                owner_id=current_user.id,
                name=template_contact.name,
                persona_prompt=template_contact.persona_prompt,
                specialty_tags=template_contact.specialty_tags,
                voice_id=template_contact.voice_id,
                avatar_emoji=template_contact.avatar_emoji,
                avatar_url=template_contact.avatar_url,
                is_template=False,
            )
            db.add(personal)
            await db.flush()

        participants.append({"id": str(personal.id), "type": "contact"})

    chat = Chat(
        owner_id=current_user.id,
        chat_type=ChatType.group,
        name=tpl["name"],
        participants=participants,
    )
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return chat
