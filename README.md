# AI Contacts — Backend API

FastAPI backend for the AI Contacts app. Phase 1 MVP.

## Quick Start

### 1. Clone & configure
```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 2. Start database & Redis
```bash
docker-compose up db redis -d
```

### 3. Install dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the POC terminal script first (Week 1 task)
```bash
python scripts/poc_terminal.py
```
This validates Claude + ElevenLabs before any UI work.

### 5. Run migrations
```bash
alembic upgrade head
```

### 6. Seed contact templates
```bash
python scripts/seed_contacts.py
```

### 7. Start the API
```bash
uvicorn app.main:app --reload
```

API docs at: http://localhost:8000/docs

---

## Project Structure

```
app/
├── api/
│   ├── deps.py              # JWT auth dependency
│   └── routes/
│       ├── auth.py          # Register / login
│       ├── contacts.py      # AI contacts CRUD
│       ├── chats.py         # Direct + group chats
│       ├── messages.py      # Send message, voice note
│       └── calls.py         # Tavus video call sessions
├── core/
│   ├── config.py            # All env vars via pydantic-settings
│   ├── security.py          # JWT creation + verification
│   └── websocket_manager.py # Per-user WebSocket connections
├── db/
│   └── session.py           # Async SQLAlchemy engine + Base
├── models/                  # SQLAlchemy ORM models
├── schemas/                 # Pydantic request/response schemas
├── services/
│   ├── ai/
│   │   ├── claude_service.py    # Claude streaming + memory extraction
│   │   └── persona_service.py   # Persona prompt generation
│   ├── memory/
│   │   └── memory_service.py    # Mem0 / pgvector memory read+write
│   ├── voice/
│   │   ├── stt_service.py       # Whisper transcription
│   │   ├── tts_service.py       # ElevenLabs TTS
│   │   └── storage_service.py   # S3/R2 upload + presigned URLs
│   └── video/
│       └── tavus_service.py     # Tavus CVI session creation
├── orchestration/           # Phase 2 — group chat multi-agent (LangGraph)
├── workers/                 # Celery background tasks
└── main.py                  # FastAPI app, CORS, WebSocket endpoint

scripts/
├── poc_terminal.py          # Week 1 validation — run this first
└── seed_contacts.py         # Seed 3 Phase 1 contact templates
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/v1/auth/register | Register new user |
| POST | /api/v1/auth/login | Login |
| GET | /api/v1/contacts/ | List contacts + templates |
| POST | /api/v1/contacts/ | Create custom contact |
| GET | /api/v1/contacts/templates | List system templates |
| GET | /api/v1/chats/ | List user's chats |
| POST | /api/v1/chats/direct | Start a direct chat |
| POST | /api/v1/chats/group | Start a group chat |
| GET | /api/v1/messages/{chat_id} | Get chat messages |
| POST | /api/v1/messages/ | Send message (text or voice) |
| POST | /api/v1/calls/video/{contact_id} | Start video call |

## WebSocket

Connect at: `ws://localhost:8000/ws/{user_id}?token=<jwt>`

Events pushed to client:
- `message` — new agent response
- `typing` — agent typing indicator (Phase 2)
- `call_invite` — incoming call (Phase 2)
# project-interface-melo-backend
