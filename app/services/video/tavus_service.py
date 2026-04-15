import httpx
from app.core.config import settings

TAVUS_BASE = "https://tavusapi.com/v2"


async def create_tavus_session(
    contact_name: str,
    persona_prompt: str,
    avatar_id: str | None = None,
    voice_id: str | None = None,
) -> dict:
    """
    Create a Tavus Conversational Video Interface session.
    Returns session URL and token for the client to render.
    """
    payload = {
        "replica_id": avatar_id or "r79e1c033f",  # default Tavus replica
        "persona_id": "p5317866",                  # default Tavus persona
        "conversation_name": f"Call with {contact_name}",
        "conversational_context": persona_prompt,
        "properties": {
            "max_call_duration": 3600,
            "participant_left_timeout": 30,
            "enable_recording": True,
        },
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{TAVUS_BASE}/conversations",
            headers={
                "x-api-key": settings.TAVUS_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    return {
        "conversation_id": data.get("conversation_id"),
        "conversation_url": data.get("conversation_url"),
        "status": data.get("status"),
    }


async def end_tavus_session(conversation_id: str) -> dict:
    """End a Tavus session and return transcript."""
    async with httpx.AsyncClient(timeout=15) as client:
        await client.delete(
            f"{TAVUS_BASE}/conversations/{conversation_id}",
            headers={"x-api-key": settings.TAVUS_API_KEY},
        )
    return {"status": "ended", "conversation_id": conversation_id}
