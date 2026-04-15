from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List
import json


class Settings(BaseSettings):
    APP_NAME: str = "AI Contacts"
    SECRET_KEY: str = "dev-secret-key"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    APP_BASE_URL: str = "http://localhost:8000"
    APP_DEEP_LINK_SCHEME: str = "aicontacts"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/ai_contacts"
    SYNC_DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/ai_contacts"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Email
    RESEND_API_KEY: str = ""
    EMAIL_FROM_ADDRESS: str = "noreply@yourdomain.com"
    EMAIL_FROM_NAME: str = "AI Contacts"

    # OAuth
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    MICROSOFT_CLIENT_ID: str = ""
    MICROSOFT_CLIENT_SECRET: str = ""
    YAHOO_CLIENT_ID: str = ""
    YAHOO_CLIENT_SECRET: str = ""

    # LLM — Claude (primary)
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"

    # LLM — Qwen via OpenRouter (cost fallback after CLAUDE_MESSAGE_LIMIT)
    OPENROUTER_API_KEY: str = ""

    # LLM — Ollama (dev fallback, free, local)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # OpenAI (Whisper API fallback only)
    OPENAI_API_KEY: str = ""

    # ElevenLabs (TTS fallback only)
    ELEVENLABS_API_KEY: str = ""

    # Tavus (video avatar, Phase 3)
    TAVUS_API_KEY: str = ""

    # Storage
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "ai-contacts-media"
    S3_ENDPOINT_URL: str = ""

    # Self-hosted services
    WHISPER_SERVICE_URL: str = "http://localhost:8001"
    KOKORO_SERVICE_URL: str = "http://localhost:8880"
    VOXTRAL_SERVICE_URL: str = ""
    LIVEKIT_URL: str = "ws://localhost:7880"
    LIVEKIT_API_KEY: str = "devkey"
    LIVEKIT_API_SECRET: str = "secret"

    # App settings
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8081"]
    MAX_MEMORIES_PER_CONTEXT: int = 10
    GROUP_CHAT_MAX_AGENTS: int = 6
    AGENT_RESPONSE_STAGGER_MS: int = 2000

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v):
        """
        Accepts both formats in .env:
          ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8081
          ALLOWED_ORIGINS=["http://localhost:3000","http://localhost:8081"]
        """
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                return json.loads(v)
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()