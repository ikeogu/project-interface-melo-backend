from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.chat import Chat, ChatType
from app.models.contact import Contact
from app.schemas.chat import ChatCreate, GroupChatCreate, ChatOut
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
