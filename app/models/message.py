import uuid
from sqlalchemy import String, DateTime, JSON, Enum as SAEnum, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import enum
from app.db.session import Base


class SenderType(str, enum.Enum):
    user = "user"
    agent = "agent"


class ContentType(str, enum.Enum):
    text = "text"
    voice = "voice"
    video = "video"


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chats.id"), nullable=False)
    sender_type: Mapped[SenderType] = mapped_column(SAEnum(SenderType), nullable=False)
    sender_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    content_type: Mapped[ContentType] = mapped_column(SAEnum(ContentType), default=ContentType.text)
    text_content: Mapped[str | None] = mapped_column(Text)
    media_url: Mapped[str | None] = mapped_column(String(500))
    transcription: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    chat = relationship("Chat", back_populates="messages")