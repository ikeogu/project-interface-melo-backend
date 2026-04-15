from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import List
from app.models.chat import ChatType


class ChatCreate(BaseModel):
    contact_id: UUID  # for direct chats
    chat_type: ChatType = ChatType.direct

class GroupChatCreate(BaseModel):
    name: str
    contact_ids: List[UUID]

class ChatOut(BaseModel):
    id: UUID
    chat_type: ChatType
    name: str | None
    participants: list
    created_at: datetime
    last_message_at: datetime | None

    class Config:
        from_attributes = True
