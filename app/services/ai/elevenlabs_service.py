from elevenlabs.client import ElevenLabs
from app.core.config import settings

class ElevenLabsService:
    def __init__(self):
        if not settings.ELEVENLABS_API_KEY:
            raise Exception("ELEVENLABS_API_KEY not set")

        self.client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        transcript = self.client.speech_to_text.convert(
            file=audio_bytes,
            model_id="scribe_v1"
        )

        return transcript.text