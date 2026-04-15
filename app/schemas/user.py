from pydantic import BaseModel, EmailStr
from uuid import UUID
from datetime import datetime
from app.models.user import AuthProvider


class UserOut(BaseModel):
    id: UUID
    email: EmailStr
    display_name: str
    avatar_url: str | None
    auth_provider: AuthProvider
    email_verified: bool
    is_profile_complete: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut
