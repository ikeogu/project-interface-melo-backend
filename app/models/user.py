import uuid
from sqlalchemy import String, DateTime, JSON, Boolean, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import enum
from app.db.session import Base


class AuthProvider(str, enum.Enum):
    email = "email"       # email + OTP
    google = "google"     # Google OAuth
    microsoft = "microsoft"  # Outlook / Microsoft OAuth
    yahoo = "yahoo"       # Yahoo OAuth


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    avatar_url: Mapped[str | None] = mapped_column(String(500))

    # Auth provider
    auth_provider: Mapped[AuthProvider] = mapped_column(SAEnum(AuthProvider), default=AuthProvider.email)
    oauth_provider_id: Mapped[str | None] = mapped_column(String(255))  # provider's user ID
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    is_profile_complete: Mapped[bool] = mapped_column(Boolean, default=False)

    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime)

    chats = relationship("Chat", back_populates="owner")
    memories = relationship("Memory", back_populates="user")
