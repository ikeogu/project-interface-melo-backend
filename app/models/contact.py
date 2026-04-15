import uuid
from sqlalchemy import String, DateTime, Boolean, JSON, Text, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime
import enum
from app.db.session import Base


class MemoryScope(str, enum.Enum):
    personal = "personal"
    shared = "shared"


class Contact(Base):
    __tablename__ = "contacts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)  # null = system template
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    persona_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    specialty_tags: Mapped[list] = mapped_column(JSON, default=list)
    voice_id: Mapped[str | None] = mapped_column(String(100))        # ElevenLabs voice ID
    avatar_id: Mapped[str | None] = mapped_column(String(100))       # Tavus avatar ID
    avatar_url: Mapped[str | None] = mapped_column(String(500))      # Static avatar image
    avatar_emoji: Mapped[str | None] = mapped_column(String(10))     # Fallback emoji/initials
    is_template: Mapped[bool] = mapped_column(Boolean, default=False)
    memory_scope: Mapped[MemoryScope] = mapped_column(SAEnum(MemoryScope), default=MemoryScope.personal)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    memories = relationship("Memory", back_populates="contact")
