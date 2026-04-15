from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from app.models.message import SenderType, ContentType


class MessageCreate(BaseModel):
    chat_id: UUID
    content_type: ContentType = ContentType.text
    text_content: str | None = None
    media_url: str | None = None  # for voice notes uploaded to S3

class MessageOut(BaseModel):
    id: UUID
    chat_id: UUID
    sender_type: SenderType
    sender_id: UUID
    content_type: ContentType
    text_content: str | None
    media_url: str | None
    transcription: str | None
    created_at: datetime

    class Config:
        from_attributes = True

class WSMessage(BaseModel):
    """WebSocket message envelope"""
    type: str  # "message" | "typing" | "call_invite" | "error"
    payload: dict
