"""
Email sending via Resend (https://resend.com).
Resend is the best choice here:
  - Free tier: 3,000 emails/month
  - Simple API, great deliverability
  - Works perfectly for transactional OTP emails
  - ~$0.0008/email after free tier

Alternative: SendGrid (more complex), Mailgun, AWS SES
"""
import httpx
from app.core.config import settings

RESEND_BASE = "https://api.resend.com"


async def send_otp_email(to_email: str, otp_code: str, display_name: str = "") -> bool:
    """Send OTP verification email. Returns True if sent successfully."""

    name_greeting = f"Hi {display_name}," if display_name else "Hi there,"

    html_body = f"""
    <div style="font-family: sans-serif; max-width: 480px; margin: 0 auto; padding: 32px;">
      <h2 style="color: #1a1a1a; margin-bottom: 8px;">Your verification code</h2>
      <p style="color: #555; margin-bottom: 24px;">{name_greeting} use the code below to sign in to AI Contacts.</p>

      <div style="background: #f5f5f5; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 24px;">
        <span style="font-size: 36px; font-weight: 700; letter-spacing: 8px; color: #1a1a1a;">{otp_code}</span>
      </div>

      <p style="color: #888; font-size: 13px;">This code expires in 10 minutes. If you didn't request this, ignore this email.</p>
    </div>
    """

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{RESEND_BASE}/emails",
            headers={
                "Authorization": f"Bearer {settings.RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": f"{settings.EMAIL_FROM_NAME} <{settings.EMAIL_FROM_ADDRESS}>",
                "to": [to_email],
                "subject": f"{otp_code} — your AI Contacts code",
                "html": html_body,
            },
        )
        return r.status_code == 200


async def send_welcome_email(to_email: str, display_name: str) -> bool:
    """Send welcome email after profile is complete."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{RESEND_BASE}/emails",
            headers={
                "Authorization": f"Bearer {settings.RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": f"{settings.EMAIL_FROM_NAME} <{settings.EMAIL_FROM_ADDRESS}>",
                "to": [to_email],
                "subject": f"Welcome to AI Contacts, {display_name}",
                "html": f"""
                <div style="font-family: sans-serif; max-width: 480px; margin: 0 auto; padding: 32px;">
                  <h2>Welcome, {display_name}!</h2>
                  <p>Your AI Contacts are ready. Alex, your personal assistant, is waiting to help.</p>
                  <p>Open the app to start your first conversation.</p>
                </div>
                """,
            },
        )
        return r.status_code == 200
