from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import List, Optional


class ContactCreate(BaseModel):
    name: str
    persona_prompt: str | None = None
    personality_description: str | None = None  # plain english -> auto-generates persona_prompt
    specialty_tags: List[str] = []
    voice_id: str | None = None
    avatar_emoji: str | None = None

class ContactOut(BaseModel):
    id: UUID
    name: str
    persona_prompt: str
    specialty_tags: List[str]
    voice_id: str | None
    avatar_url: str | None
    avatar_emoji: str | None
    is_template: bool
    created_at: datetime

    class Config:
        from_attributes = True
