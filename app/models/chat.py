import uuid
from sqlalchemy import String, DateTime, JSON, Enum as SAEnum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import enum
from app.db.session import Base


class ChatType(str, enum.Enum):
    direct = "direct"
    group = "group"
    mixed = "mixed"


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    chat_type: Mapped[ChatType] = mapped_column(SAEnum(ChatType), default=ChatType.direct)
    name: Mapped[str | None] = mapped_column(String(100))
    # participants = list of {id, type: "user"|"contact"}
    participants: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime)

    owner = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", order_by="Message.created_at")
