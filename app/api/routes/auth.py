"""
Complete email + OAuth auth flow.

Email path:
  POST /auth/send-otp         → sends 6-digit code to email
  POST /auth/verify-otp       → verifies code, returns JWT + is_new_user
  POST /auth/complete-profile → (new users) set name + avatar

OAuth path (Google / Microsoft / Yahoo):
  GET  /auth/oauth/{provider}/url       → get redirect URL for provider
  GET  /auth/oauth/{provider}/callback  → handles redirect, returns JWT

Shared:
  GET  /auth/me               → current user profile
  GET  /auth/avatar-upload-url → presigned S3 URL for avatar upload
  POST /auth/logout           → client-side token deletion
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from datetime import datetime
from app.db.session import get_db
from app.models.user import User, AuthProvider
from app.schemas.user import UserOut, TokenOut
from app.core.security import create_access_token
from app.core.config import settings
from app.api.deps import get_current_user
from app.services.voice.storage_service import get_presigned_upload_url

router = APIRouter(prefix="/auth", tags=["auth"])

SUPPORTED_PROVIDERS = ["google", "microsoft", "yahoo"]


# ── Schemas ────────────────────────────────────────────────────────────────────

class SendOTPRequest(BaseModel):
    email: EmailStr

class VerifyOTPRequest(BaseModel):
    email: EmailStr
    code: str

class CompleteProfileRequest(BaseModel):
    display_name: str
    avatar_url: str | None = None

class VerifyOTPResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut
    is_new_user: bool
    is_profile_complete: bool

class AvatarUploadResponse(BaseModel):
    upload_url: str
    public_url: str


# ── Email OTP ─────────────────────────────────────────────────────────────────

@router.post("/send-otp")
async def send_otp(payload: SendOTPRequest):
    """
    Step 1 (email path): Send a 6-digit OTP to the user's email.
    Works for both new and existing users.
    """
    from app.services.auth.otp_service import create_otp
    from app.services.auth.email_service import send_otp_email

    code = await create_otp(payload.email)

    if settings.ENVIRONMENT == "development":
        # Skip real email in dev — print to console
        print(f"\n[DEV] OTP for {payload.email}: {code}\n")
        return {"message": f"Dev mode — OTP printed to console"}

    sent = await send_otp_email(to_email=payload.email, otp_code=code)
    if not sent:
        raise HTTPException(status_code=500, detail="Failed to send email. Try again.")

    return {"message": "Verification code sent to your email"}


@router.post("/verify-otp", response_model=VerifyOTPResponse)
async def verify_otp(payload: VerifyOTPRequest, db: AsyncSession = Depends(get_db)):
    """
    Step 2 (email path): Verify OTP code.
    Creates account if first time, updates last_login_at if returning.
    """
    from app.services.auth.otp_service import verify_otp as check_otp

    valid, error = await check_otp(payload.email, payload.code)
    if not valid:
        raise HTTPException(status_code=400, detail=error)

    return await _get_or_create_user(
        db=db,
        email=payload.email,
        provider=AuthProvider.email,
        email_verified=True,
    )


# ── OAuth ──────────────────────────────────────────────────────────────────────

@router.get("/oauth/{provider}/url")
async def get_oauth_url(provider: str):
    """
    Step 1 (OAuth path): Get the provider's login URL.
    React Native opens this in a WebBrowser / InAppBrowser.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Provider must be one of: {SUPPORTED_PROVIDERS}")

    from app.services.auth.oauth_service import get_oauth_redirect_url
    url = get_oauth_redirect_url(provider)
    return {"url": url, "provider": provider}


@router.get("/oauth/{provider}/callback")
async def oauth_callback(
    provider: str,
    code: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Step 2 (OAuth path): Provider redirects here after user logs in.
    Exchanges code for user info, creates/finds user, returns JWT.

    NOTE: In a mobile app the redirect goes back to the app via deep link
    (e.g. aicontacts://auth/callback?code=...). The app then calls
    POST /auth/oauth/{provider}/exchange with the code instead of this
    redirect endpoint. Both patterns work — this one is simpler for web.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=400, detail="Invalid provider")

    from app.services.auth.oauth_service import exchange_code_for_user_info
    try:
        profile = await exchange_code_for_user_info(provider, code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OAuth failed: {str(e)}")

    if not profile.get("email"):
        raise HTTPException(status_code=400, detail="Could not retrieve email from provider")

    provider_enum = AuthProvider(provider)
    result = await _get_or_create_user(
        db=db,
        email=profile["email"],
        provider=provider_enum,
        email_verified=profile.get("email_verified", True),
        display_name=profile.get("name", ""),
        avatar_url=profile.get("avatar_url"),
        oauth_provider_id=profile.get("provider_id"),
    )

    # For mobile deep-link flow: redirect to app with token
    if settings.ENVIRONMENT != "development":
        deep_link = (
            f"{settings.APP_DEEP_LINK_SCHEME}://auth/callback"
            f"?token={result.access_token}"
            f"&is_new_user={str(result.is_new_user).lower()}"
        )
        return RedirectResponse(url=deep_link)

    return result


@router.post("/oauth/{provider}/exchange", response_model=VerifyOTPResponse)
async def oauth_exchange(
    provider: str,
    code: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Mobile alternative to the callback redirect.
    React Native app gets the OAuth code via deep link,
    then calls this endpoint to exchange it for a JWT.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=400, detail="Invalid provider")

    from app.services.auth.oauth_service import exchange_code_for_user_info
    try:
        profile = await exchange_code_for_user_info(provider, code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OAuth exchange failed: {str(e)}")

    return await _get_or_create_user(
        db=db,
        email=profile["email"],
        provider=AuthProvider(provider),
        email_verified=True,
        display_name=profile.get("name", ""),
        avatar_url=profile.get("avatar_url"),
        oauth_provider_id=profile.get("provider_id"),
    )


# ── Profile ────────────────────────────────────────────────────────────────────

@router.post("/complete-profile", response_model=UserOut)
async def complete_profile(
    payload: CompleteProfileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    New users only: Set display name + optional avatar.
    Also seeds Alex (Personal Assistant) as their first contact.
    Marks is_profile_complete = True.
    """
    current_user.display_name = payload.display_name
    current_user.is_profile_complete = True
    if payload.avatar_url:
        current_user.avatar_url = payload.avatar_url
    await db.commit()
    await db.refresh(current_user)

    await _seed_default_contact(current_user.id, db)

    from app.services.auth.email_service import send_welcome_email
    if settings.ENVIRONMENT != "development":
        await send_welcome_email(current_user.email, payload.display_name)

    return UserOut.model_validate(current_user)


@router.get("/avatar-upload-url", response_model=AvatarUploadResponse)
async def get_avatar_upload_url(current_user: User = Depends(get_current_user)):
    """Presigned S3 URL — client uploads image directly, saves the public_url."""
    result = get_presigned_upload_url(f"avatar-{current_user.id}.jpg")
    return AvatarUploadResponse(
        upload_url=result["upload_url"],
        public_url=result["public_url"],
    )


@router.get("/me", response_model=UserOut)
async def get_me(current_user: User = Depends(get_current_user)):
    """Restore session on app launch. Returns 401 if token expired."""
    return UserOut.model_validate(current_user)


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    return {"message": "Logged out — delete token on client"}


# ── Shared helper ──────────────────────────────────────────────────────────────

async def _get_or_create_user(
    db: AsyncSession,
    email: str,
    provider: AuthProvider,
    email_verified: bool = False,
    display_name: str = "",
    avatar_url: str | None = None,
    oauth_provider_id: str | None = None,
) -> VerifyOTPResponse:
    """
    Find existing user by email or create new one.
    Handles the case where someone tries to log in with OAuth after
    previously registering with email (merges accounts by email).
    """
    result = await db.execute(select(User).where(User.email == email.lower()))
    user = result.scalar_one_or_none()
    is_new = user is None

    if is_new:
        user = User(
            email=email.lower(),
            display_name=display_name,
            avatar_url=avatar_url,
            auth_provider=provider,
            oauth_provider_id=oauth_provider_id,
            email_verified=email_verified,
            is_profile_complete=bool(display_name),  # OAuth gives us name already
        )
        db.add(user)
    else:
        # Returning user — update login timestamp and fill any missing fields
        user.last_login_at = datetime.utcnow()
        if not user.avatar_url and avatar_url:
            user.avatar_url = avatar_url
        if not user.display_name and display_name:
            user.display_name = display_name
        if oauth_provider_id and not user.oauth_provider_id:
            user.oauth_provider_id = oauth_provider_id

    await db.commit()
    await db.refresh(user)

    # Auto-seed Alex for new OAuth users who skip the profile step
    if is_new and display_name:
        await _seed_default_contact(user.id, db)

    token = create_access_token({"sub": str(user.id)})
    return VerifyOTPResponse(
        access_token=token,
        user=UserOut.model_validate(user),
        is_new_user=is_new,
        is_profile_complete=user.is_profile_complete,
    )


async def _seed_default_contact(user_id, db: AsyncSession):
    """Give every new user their first contact (Alex, Personal Assistant)."""
    from app.models.contact import Contact
    result = await db.execute(
        select(Contact).where(Contact.is_template == True, Contact.name == "Alex")
    )
    template = result.scalar_one_or_none()
    if not template:
        return
    personal_alex = Contact(
        owner_id=user_id,
        name=template.name,
        persona_prompt=template.persona_prompt,
        specialty_tags=template.specialty_tags,
        voice_id=template.voice_id,
        avatar_emoji=template.avatar_emoji,
        is_template=False,
    )
    db.add(personal_alex)
    await db.commit()
