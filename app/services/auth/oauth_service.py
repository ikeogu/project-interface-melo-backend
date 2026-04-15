"""
OAuth 2.0 for Google, Microsoft (Outlook), and Yahoo.

Flow for all three providers:
  1. Client calls GET /auth/oauth/{provider}/url  → gets redirect URL
  2. User is redirected to provider login page
  3. Provider redirects back to GET /auth/oauth/{provider}/callback?code=...
  4. Backend exchanges code for tokens, fetches user email
  5. User is created/found, JWT is returned

Provider docs:
  Google:    https://developers.google.com/identity/protocols/oauth2
  Microsoft: https://learn.microsoft.com/en-us/azure/active-directory/develop/v2-oauth2-auth-code-flow
  Yahoo:     https://developer.yahoo.com/oauth2/guide/
"""
import httpx
from urllib.parse import urlencode
from app.core.config import settings


PROVIDERS = {
    "google": {
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://www.googleapis.com/oauth2/v3/userinfo",
        "client_id": lambda: settings.GOOGLE_CLIENT_ID,
        "client_secret": lambda: settings.GOOGLE_CLIENT_SECRET,
        "scope": "openid email profile",
    },
    "microsoft": {
        "auth_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "userinfo_url": "https://graph.microsoft.com/v1.0/me",
        "client_id": lambda: settings.MICROSOFT_CLIENT_ID,
        "client_secret": lambda: settings.MICROSOFT_CLIENT_SECRET,
        "scope": "openid email profile User.Read",
    },
    "yahoo": {
        "auth_url": "https://api.login.yahoo.com/oauth2/request_auth",
        "token_url": "https://api.login.yahoo.com/oauth2/get_token",
        "userinfo_url": "https://api.login.yahoo.com/openid/v1/userinfo",
        "client_id": lambda: settings.YAHOO_CLIENT_ID,
        "client_secret": lambda: settings.YAHOO_CLIENT_SECRET,
        "scope": "openid email profile",
    },
}


def get_oauth_redirect_url(provider: str) -> str:
    """Build the OAuth authorization URL to redirect the user to."""
    cfg = PROVIDERS[provider]
    callback_url = f"{settings.APP_BASE_URL}/api/v1/auth/oauth/{provider}/callback"

    params = {
        "client_id": cfg["client_id"](),
        "redirect_uri": callback_url,
        "response_type": "code",
        "scope": cfg["scope"],
        "access_type": "offline",  # Google-specific, ignored by others
        "prompt": "select_account",
    }
    return f"{cfg['auth_url']}?{urlencode(params)}"


async def exchange_code_for_user_info(provider: str, code: str) -> dict:
    """
    Exchange OAuth code for access token, then fetch user profile.
    Returns dict with: email, name, avatar_url, provider_id
    """
    cfg = PROVIDERS[provider]
    callback_url = f"{settings.APP_BASE_URL}/api/v1/auth/oauth/{provider}/callback"

    # Step 1: Exchange code for token
    async with httpx.AsyncClient(timeout=15) as client:
        token_r = await client.post(
            cfg["token_url"],
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": callback_url,
                "client_id": cfg["client_id"](),
                "client_secret": cfg["client_secret"](),
            },
            headers={"Accept": "application/json"},
        )
        token_r.raise_for_status()
        tokens = token_r.json()
        access_token = tokens["access_token"]

        # Step 2: Fetch user profile
        profile_r = await client.get(
            cfg["userinfo_url"],
            headers={"Authorization": f"Bearer {access_token}"},
        )
        profile_r.raise_for_status()
        profile = profile_r.json()

    # Normalise across providers
    return _normalise_profile(provider, profile)


def _normalise_profile(provider: str, profile: dict) -> dict:
    """Map provider-specific profile fields to a common format."""
    if provider == "google":
        return {
            "email": profile.get("email", ""),
            "name": profile.get("name", ""),
            "avatar_url": profile.get("picture"),
            "provider_id": profile.get("sub"),
            "email_verified": profile.get("email_verified", False),
        }
    elif provider == "microsoft":
        return {
            # Microsoft Graph uses 'mail' or 'userPrincipalName'
            "email": profile.get("mail") or profile.get("userPrincipalName", ""),
            "name": profile.get("displayName", ""),
            "avatar_url": None,  # needs separate Graph call for photo
            "provider_id": profile.get("id"),
            "email_verified": True,  # Microsoft verifies emails
        }
    elif provider == "yahoo":
        return {
            "email": profile.get("email", ""),
            "name": profile.get("name", ""),
            "avatar_url": profile.get("picture"),
            "provider_id": profile.get("sub"),
            "email_verified": profile.get("email_verified", False),
        }
    return {}
