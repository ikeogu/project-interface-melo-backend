"""
Contacts routes — handles:
- Listing user's own contacts + available templates
- Adding a template to your contacts (one tap from UI)
- Creating a fully custom contact
- Deleting a contact
- Getting a single contact detail
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from pydantic import BaseModel
from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.contact import Contact
from app.schemas.contact import ContactCreate, ContactOut
from app.services.ai.claude_service import generate_persona_prompt
from uuid import UUID

router = APIRouter(prefix="/contacts", tags=["contacts"])


class AddTemplateRequest(BaseModel):
    template_id: UUID


# ── User's contacts ────────────────────────────────────────────────────────────

@router.get("/", response_model=list[ContactOut])
async def list_my_contacts(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns only this user's own contacts (not templates).
    This populates the YOUR CONTACTS section in the UI.
    """
    result = await db.execute(
        select(Contact)
        .where(Contact.owner_id == current_user.id)
        .order_by(Contact.created_at.asc())
    )
    return result.scalars().all()


@router.get("/templates", response_model=list[ContactOut])
async def list_templates(db: AsyncSession = Depends(get_db)):
    """
    Returns all system templates.
    This populates the TEMPLATES — TAP TO ADD section in the UI.
    """
    result = await db.execute(
        select(Contact).where(Contact.is_template == True).order_by(Contact.name)
    )
    return result.scalars().all()


@router.post("/add-template", response_model=ContactOut)
async def add_template_to_contacts(
    payload: AddTemplateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    One-tap: user taps '+ Add' on a template → creates a personal copy.
    This is the primary way users populate their contacts list.
    """
    template_result = await db.execute(
        select(Contact).where(Contact.id == payload.template_id, Contact.is_template == True)
    )
    template = template_result.scalar_one_or_none()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Check if user already has this contact
    existing = await db.execute(
        select(Contact).where(
            Contact.owner_id == current_user.id,
            Contact.name == template.name,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="You already have this contact")

    personal_copy = Contact(
        owner_id=current_user.id,
        name=template.name,
        persona_prompt=template.persona_prompt,
        specialty_tags=template.specialty_tags,
        voice_id=template.voice_id,
        avatar_emoji=template.avatar_emoji,
        avatar_url=template.avatar_url,
        is_template=False,
    )
    db.add(personal_copy)
    await db.commit()
    await db.refresh(personal_copy)
    return personal_copy


@router.post("/", response_model=ContactOut)
async def create_custom_contact(
    payload: ContactCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a fully custom contact from the 'Create a Contact' screen.
    If only personality_description is given, Claude auto-generates the persona prompt.
    """
    persona_prompt = payload.persona_prompt
    if not persona_prompt and payload.personality_description:
        persona_prompt = await generate_persona_prompt(
            payload.name, payload.personality_description
        )
    elif not persona_prompt:
        raise HTTPException(
            status_code=400,
            detail="Provide either persona_prompt or personality_description"
        )

    contact = Contact(
        owner_id=current_user.id,
        name=payload.name,
        persona_prompt=persona_prompt,
        specialty_tags=payload.specialty_tags,
        voice_id=payload.voice_id,
        avatar_emoji=payload.avatar_emoji,
        is_template=False,
    )
    db.add(contact)
    await db.commit()
    await db.refresh(contact)
    return contact


@router.get("/{contact_id}", response_model=ContactOut)
async def get_contact(
    contact_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Contact).where(
            Contact.id == contact_id,
            Contact.owner_id == current_user.id,
        )
    )
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    return contact


@router.delete("/{contact_id}", status_code=204)
async def delete_contact(
    contact_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Contact).where(
            Contact.id == contact_id,
            Contact.owner_id == current_user.id
        )
    )
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    await db.delete(contact)
    await db.commit()
