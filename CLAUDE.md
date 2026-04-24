# Aura — AI Contacts App
## Full Project Context for Claude Code

---

## What this app is

**Aura** is a WhatsApp-style messaging app where the contacts are AI agents.
Every contact is an AI persona with a name, avatar, personality, expertise, and voice.
Users interact with their AI contacts exactly how they would with real people:
text chat, voice notes, and video calls.

**Core product thesis:** AI is most useful when it feels like a person you know, not a tool you operate.

---

## Tech stack

### Backend (built and deployed on Railway)
- **FastAPI (Python)** — REST API + WebSockets
- **PostgreSQL via Supabase** — primary database
- **Redis via Upstash** — OTP storage
- **Supabase Storage** — voice note files
- **Anthropic Claude API** — primary LLM (messages 1–20 per chat)
- **Qwen via OpenRouter** — cost fallback (messages 21+)
- **Ollama** — local dev fallback (free, no API cost)
- **OpenAI Whisper** — speech-to-text for voice notes
- **ElevenLabs** — text-to-speech for agent voice responses
- **LiveKit** — WebRTC for audio/video calls (Phase 3)

### Frontend (React Native — what Claude Code works on)
- **React Native + Expo SDK 54**
- **React Navigation v7** (Stack + Bottom Tabs) — NOT Expo Router
- **Zustand** — state management (no Redux, no Context)
- **Axios** — HTTP client with JWT interceptor
- **expo-audio** — recording + playback (NOT expo-av — deprecated in SDK 54)
- **expo-secure-store** — JWT token storage
- **Entry point:** `index.js` → `App.js` → `AppNavigator`
- **package.json `main`:** `"index.js"` (not expo-router/entry)

---

## Project file structure

```
ai-contacts-app/
├── index.js                           # Entry: registers App component
├── App.js                             # GestureHandlerRootView + SafeAreaProvider + AppNavigator
├── src/
│   ├── api/
│   │   ├── client.js                  # Axios instance, JWT interceptor, 401 handler
│   │   ├── auth.js                    # sendOtp, verifyOtp, completeProfile, getMe, logout
│   │   ├── contacts.js                # getMyContacts, getTemplates, addTemplate, createContact
│   │   ├── chats.js                   # getChats, createDirectChat, createGroupChat, getMessages, sendMessage
│   │   └── voice.js                   # uploadVoiceNote, sendVoiceNote
│   ├── store/
│   │   ├── authStore.js               # user, token, isAuthenticated, initialize, setAuth, updateUser, logout
│   │   ├── contactsStore.js           # contacts[], templates[], fetchContacts, addFromTemplate, createContact
│   │   └── chatsStore.js              # chats[], messages{chat_id: []}, sendMessage (optimistic), appendMessage
│   ├── screens/
│   │   ├── SplashScreen.jsx           # Animated logo, 2.2s, deep green background
│   │   ├── EmailScreen.jsx            # Email input + social login buttons
│   │   ├── OTPScreen.jsx              # 6 individual digit boxes, auto-advance, resend countdown
│   │   ├── ProfileSetupScreen.jsx     # Name input, live avatar preview
│   │   ├── ContactsScreen.jsx         # SectionList: YOUR CONTACTS + TEMPLATES sections
│   │   ├── ChatsScreen.jsx            # FlatList of chat rows sorted by last_message_at
│   │   ├── ChatScreen.jsx             # Messages + WebSocket + text + voice input
│   │   ├── CreateContactScreen.jsx    # Emoji picker, name, description, voice selector
│   │   ├── VideoCallScreen.jsx        # Full screen call UI with controls
│   │   └── ProfileScreen.jsx          # User info, settings, sign out
│   ├── components/
│   │   ├── Avatar.jsx                 # Circle with initials/emoji/image, auto bg color by name
│   │   ├── ContactRow.jsx             # Contact list row, accepts rightAction prop
│   │   ├── ChatRow.jsx                # Chat list row with name, subtitle, timestamp
│   │   ├── MessageBubble.jsx          # Text bubble: user=right/green, agent=left/white
│   │   ├── VoiceRecorder.jsx          # onPressIn records, onPressOut stops — pulse animation
│   │   └── VoiceNoteBubble.jsx        # Waveform bars, play/pause, progress, transcription
│   ├── navigation/
│   │   └── AppNavigator.jsx           # Splash → Auth stack | Main tabs decision
│   ├── utils/
│   │   ├── websocket.js               # wsConnect(userId, onMessage), wsDisconnect, auto-reconnect
│   │   └── format.js                  # formatChatTime, formatMessageTime, getInitials, getAvatarColor
│   └── theme/
│       └── colors.js                  # Full design token palette
```

---

## Backend API reference

**Base URL:** `https://project-interface-melo-backend-production.up.railway.app/api/v1`

All protected routes require header: `Authorization: Bearer <jwt_token>`

### Auth endpoints
```
POST /auth/send-otp          { email }                      → { message }
POST /auth/verify-otp        { email, code }                → { access_token, user, is_new_user, is_profile_complete }
POST /auth/complete-profile  { display_name, avatar_url? }  → User
GET  /auth/me                                               → User
POST /auth/logout                                          → { message }

GET  /auth/oauth/{provider}/url       → { url }
POST /auth/oauth/{provider}/exchange  ?code=...  → { access_token, user, is_new_user }
```

### Contacts endpoints
```
GET    /contacts/              → Contact[]   (user's own contacts only)
GET    /contacts/templates     → Contact[]   (system templates — shown as "tap to add")
POST   /contacts/add-template  { template_id }  → Contact  (copies template to user's contacts)
POST   /contacts/              { name, personality_description, voice_id?, avatar_emoji? } → Contact
GET    /contacts/{id}          → Contact
DELETE /contacts/{id}          → 204
```

### Chats & messages
```
GET  /chats/                   → Chat[]
POST /chats/direct             { contact_id }               → Chat
POST /chats/group              { name, contact_ids[] }      → Chat
GET  /messages/{chat_id}       → Message[]
POST /messages/                { chat_id, text_content?, content_type, media_url? } → Message
```

### Voice
```
POST /voice/upload             multipart/form-data: file    → { url, key }
```

### Calls (Phase 3)
```
POST /calls/voice/{contact_id} → { room_name, token, livekit_url }
POST /calls/video/{contact_id} → { room_name, token, livekit_url }
```

### WebSocket
```
Connect: ws://BASE_URL/ws/{user_id}?token={jwt}

Messages FROM server:
  { type: "message", payload: Message }      → new agent reply
  { type: "typing",  payload: { chat_id } }  → agent is generating

Messages TO server:
  "ping"  →  server replies "pong"  (keepalive, every 30s)
```

---

## Data models

### User
```typescript
{
  id: string                  // UUID
  email: string
  display_name: string
  avatar_url: string | null
  auth_provider: "email" | "google" | "microsoft" | "yahoo"
  email_verified: boolean
  is_profile_complete: boolean
  created_at: string          // ISO 8601
}
```

### Contact
```typescript
{
  id: string                  // UUID
  owner_id: string | null     // null = system template visible to all users
  name: string                // "Alex", "Maya", "Father James"
  persona_prompt: string      // full Claude system prompt
  specialty_tags: string[]    // ["productivity", "scheduling", "research"]
  voice_id: string            // ElevenLabs/Kokoro voice ID
  avatar_url: string | null
  avatar_emoji: string | null // "A", "M", "FJ", "🤖"
  is_template: boolean
  created_at: string
}
```

### Chat
```typescript
{
  id: string
  owner_id: string
  chat_type: "direct" | "group" | "mixed"
  name: string | null
  participants: Array<{
    id: string
    type: "user" | "contact"
  }>
  created_at: string
  last_message_at: string | null
}
```

### Message
```typescript
{
  id: string
  chat_id: string
  sender_type: "user" | "agent"
  sender_id: string
  content_type: "text" | "voice" | "video"
  text_content: string | null
  media_url: string | null      // Supabase Storage URL for voice notes
  transcription: string | null  // Whisper transcript of voice note
  meta: object                  // { duration?: number, contact_name?: string }
  created_at: string

  // Frontend-only (not from API):
  pending?: boolean             // optimistic update in progress
  failed?: boolean              // send failed
}
```

---

## Design system

### Color palette
```javascript
// Primary brand
primary:             '#0D4F3C'  // deep forest green — headers, primary buttons
primaryLight:        '#1A6B54'
primaryDark:         '#082E23'

// Accent
accent:              '#25D366'  // bright green — Add button, CTAs
accentLight:         '#DCF8C6'

// Backgrounds
background:          '#FFFFFF'
backgroundSecondary: '#F5F5F5'
backgroundTertiary:  '#EBEBEB'
chatBackground:      '#ECE5DD'  // WhatsApp warm grey — used as chat screen bg

// Chat bubbles
bubbleUser:          '#DCF8C6'  // outgoing — light green
bubbleAgent:         '#FFFFFF'  // incoming — white
bubbleAgentBorder:   '#E8E8E8'

// Text
textPrimary:         '#1A1A1A'
textSecondary:       '#8A8A8A'
textTertiary:        '#BBBBBB'
textOnPrimary:       '#FFFFFF'

// UI elements
border:              '#E8E8E8'
separator:           '#EEEEEE'
inputBackground:     '#F5F5F5'

// Status
online:              '#4CAF50'
error:               '#E53935'
tabActive:           '#0D4F3C'
tabInactive:         '#9E9E9E'
```

### UI patterns
- **Chat screen background:** `#ECE5DD` warm grey, not white
- **Headers:** `primary` background, white text, white icons
- **User messages:** right-aligned, `#DCF8C6` bubble, `borderTopRightRadius: 2`
- **Agent messages:** left-aligned, white bubble, `borderTopLeftRadius: 2`, light border
- **Tab bar:** 80px height, 20px bottom padding for iPhone home indicator
- **Input bar:** white bg, `0.5px` top border, `useSafeAreaInsets().bottom` for padding
- **Avatars:** circle, initials or emoji, background color derived from name first char
- **Section headers:** 12px, 600 weight, `textSecondary`, all caps, letter-spacing 0.5

---

## Critical implementation notes

### 1. Keyboard handling in ChatScreen
```jsx
// CORRECT pattern — header has its own SafeAreaView
<View style={{ flex: 1 }}>
  <SafeAreaView edges={['top']} style={{ backgroundColor: Colors.primary }}>
    <Header />
  </SafeAreaView>
  <KeyboardAvoidingView
    style={{ flex: 1 }}
    behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    keyboardVerticalOffset={0}
  >
    <MessageList />
    <InputBar style={{ paddingBottom: Math.max(insets.bottom, 8) }} />
  </KeyboardAvoidingView>
</View>
```

### 2. Optimistic messages (chatsStore)
```javascript
// sendMessage flow:
// 1. Add tempMsg with id="temp-{timestamp}" and pending=true
// 2. Await API call
// 3. Replace temp with confirmed: messages.map(m => m.id === tempId ? confirmed : m)
// 4. Agent reply appended separately via appendMessage() from WebSocket
// NEVER: filter out temp then expect re-render — message disappears
```

### 3. expo-audio (NOT expo-av)
```javascript
// Recording
import { useAudioRecorder, AudioModule, RecordingPresets } from 'expo-audio';
const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
await AudioModule.requestRecordingPermissionsAsync();
await recorder.prepareToRecordAsync();
recorder.record();
await recorder.stop();
const uri = recorder.uri;

// Playback
import { useAudioPlayer, useAudioPlayerStatus } from 'expo-audio';
const player = useAudioPlayer({ uri: audioUrl });
const status = useAudioPlayerStatus(player);
// status.playing, status.currentTime, status.duration
player.play(); player.pause();
```

### 4. Voice recording UX
```jsx
// onPressIn starts, onPressOut stops — no hold delay
<TouchableOpacity
  onPressIn={startRecording}
  onPressOut={stopRecording}
  delayPressIn={0}
>
```

### 5. WebSocket in ChatScreen
```javascript
useEffect(() => {
  wsConnect(user.id, (data) => {
    if (data.type === 'message' && data.payload.chat_id === chat.id) {
      appendMessage(chat.id, data.payload);
    }
  });
  return () => wsDisconnect(); // cleanup on unmount
}, []);
```

### 6. message.meta not message.metadata
```javascript
// The DB column is "metadata" but Python attr is "meta" due to SQLAlchemy conflict
// API returns: message.meta = { duration: 12, contact_name: "Alex" }
const duration = message.meta?.duration;  // CORRECT
const duration = message.metadata?.duration;  // WRONG
```

---

## Auth flow

```
App launch
  ↓
SplashScreen (2.2s)
  ↓
Check SecureStore for JWT
  ├─ Valid token → GET /auth/me
  │     ├─ is_profile_complete: true  → Main tabs (Contacts/Chats/Me)
  │     └─ is_profile_complete: false → ProfileSetupScreen
  └─ No token → EmailScreen
                  ↓
               OTPScreen
                  ↓
               is_new_user && !is_profile_complete?
                  ├─ yes → ProfileSetupScreen → Main tabs
                  └─ no  → Main tabs
```

### OTP dev mode
In development, the backend prints the OTP to terminal instead of sending email.
Code is always `123456` when `ENVIRONMENT=development`.

---

## Navigation structure

```
AppNavigator (Stack)
├── [unauthenticated]
│   ├── Email
│   ├── OTP
│   └── ProfileSetup
└── [authenticated + profile complete]
    └── Main (BottomTabs)
        ├── Contacts tab (Stack)
        │   ├── ContactsList        ← default
        │   ├── Chat
        │   ├── CreateContact
        │   └── VideoCall           ← fullScreenModal
        ├── Chats tab (Stack)
        │   ├── ChatsList           ← default
        │   ├── Chat
        │   └── VideoCall           ← fullScreenModal
        └── Me tab
            └── ProfileScreen       ← no stack needed
```

---

## Pre-built contact templates (seeded in DB)

| Name | Specialty | Voice | Emoji |
|---|---|---|---|
| Alex | Personal Assistant | EXAVITQu4vr4xnSDxMaL | A |
| Maya | Wellness Coach | ThT5KcBeYPX3keUQqHPh | M |
| Father James | Pastoral Care | VR6AewLTigWG4xSOukaG | FJ |

---

## MVP scope — what's done vs what's next

### Phase 1 — DONE ✅
- Email OTP authentication
- Contacts page (your contacts + template library)
- Text chat with AI (Claude → Qwen → Ollama fallback chain)
- Real-time WebSocket message delivery
- Voice notes (record → upload → Whisper STT → Claude → ElevenLabs TTS)
- Contact memory (contacts remember facts across conversations)
- Create custom AI contact (name + description → Claude auto-generates persona)
- 3 pre-built templates: Alex, Maya, Father James
- Backend deployed on Railway
- Supabase for DB + Storage

### Phase 2 — NEXT
- Group chats (multiple AI contacts, each responds from their specialty)
- Coordinator agent (LangGraph — decides who responds, in what order)
- Staggered delivery with typing indicators (feels like real group chat)
- Mixed group chats (real user + AI contacts together)
- Cross-chat memory (PA knows what happened in the CEO group)
- Pre-built group templates (CEO Board, Trip Planning crew, etc.)

### Phase 3 — LATER
- Video calls with AI avatar (Tavus CVI)
- Audio-only AI calls (LiveKit Agent + Faster-Whisper + Voxtral)
- Emotion detection from camera (MediaPipe → feeds Claude context)
- Each contact gets a unique photorealistic face (Tavus replica)

---

## Things still to build / improve on frontend

### High priority
- [ ] Error states on all screens (network error, empty state)
- [ ] Loading skeletons for contacts and chat lists
- [ ] Typing indicator animation (animated dots, not static)
- [ ] Message timestamp grouping (show "TODAY", "YESTERDAY" dividers)
- [ ] Contact detail screen (tap contact name in chat header)
- [ ] Pull-to-refresh on contacts and chats

### Medium priority
- [ ] Message long-press menu (copy text, delete, reply-to)
- [ ] Onboarding walkthrough after first login
- [ ] Push notifications (Expo Notifications + FCM)
- [ ] Notification badge count on Chats tab
- [ ] Chat search
- [ ] Voice note amplitude waveform during recording (real bars, not fake)

### Lower priority
- [ ] Dark mode support
- [ ] Group chat creation UI flow
- [ ] Contact edit screen
- [ ] Swipe to reply on messages
- [ ] Message reactions