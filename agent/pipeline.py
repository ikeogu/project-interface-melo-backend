"""
The core call pipeline.

This is where STT, LLM, and TTS are wired together inside a LiveKit room.
The pipeline runs as a background task for the duration of each call.

Call flow per turn:
┌─────────────────────────────────────────────────────┐
│  1. LiveKit delivers user audio frames               │
│  2. VAD detects end of utterance                     │
│  3. Faster-Whisper transcribes the segment           │
│  4. LLM streams a text response                      │
│  5. TTS converts text chunks → audio as they arrive  │
│  6. Audio published back into the LiveKit room       │
│  7. Context updated, memories queued for extraction  │
└─────────────────────────────────────────────────────┘

Interruption handling:
  If the user starts speaking while the agent is speaking,
  we cancel the current TTS stream immediately and process
  the new utterance. This makes the AI feel responsive.
"""
import asyncio
import logging
from typing import AsyncGenerator

from livekit import rtc
from livekit.agents import JobContext

from agent.context import CallContext
from agent.stt import FasterWhisperSTT, Transcription
from agent.tts import SelfHostedTTS
from agent.llm import CallLLM, LLMConfig

log = logging.getLogger("pipeline")

# How long to wait after LLM finishes before checking for new user speech
TURN_TAKEOVER_PAUSE_MS = 300

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
SAMPLES_PER_FRAME = 960   # 60ms at 16kHz


class CallPipeline:
    """
    Orchestrates the full STT → LLM → TTS pipeline for one call.
    One instance per LiveKit room.
    """

    def __init__(
        self,
        ctx: JobContext,
        call_context: CallContext,
        stt: FasterWhisperSTT,
        tts: SelfHostedTTS,
        llm: CallLLM,
    ):
        self.ctx = ctx
        self.call_ctx = call_context
        self.stt = stt
        self.tts = tts
        self.llm = llm

        # Audio source — we publish this into the room
        self._audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
        self._local_track: rtc.LocalAudioTrack | None = None

        # Cancellation token for current TTS stream
        # so we can interrupt the agent mid-speech
        self._tts_task: asyncio.Task | None = None
        self._interrupted = asyncio.Event()

    async def start(self):
        """
        Main entry point. Called once when the agent joins the room.
        Publishes audio track, greets the user, then enters listen loop.
        """
        room = self.ctx.room

        # Publish our audio track so user can hear the agent
        self._local_track = rtc.LocalAudioTrack.create_audio_track(
            "agent-voice", self._audio_source
        )
        await room.local_participant.publish_track(
            self._local_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        log.info(f"[{self.call_ctx.contact_name}] Audio track published")

        # Greet the user immediately
        await self._speak_greeting()

        # Subscribe to user's audio tracks
        for participant in room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track and pub.kind == rtc.TrackKind.KIND_AUDIO:
                    await self._handle_audio_track(pub.track)

        # Listen for new participants (in case user joins after agent)
        @room.on("track_subscribed")
        def on_track(track, pub, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.ensure_future(self._handle_audio_track(track))

    async def _handle_audio_track(self, track: rtc.RemoteAudioTrack):
        """
        Process incoming audio from the user.
        Converts LiveKit audio frames to PCM, detects utterances,
        and kicks off the STT → LLM → TTS chain.
        """
        log.info(f"[{self.call_ctx.contact_name}] Subscribed to user audio")
        audio_stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)

        async def pcm_generator() -> AsyncGenerator[bytes, None]:
            async for frame_event in audio_stream:
                frame = frame_event.frame
                # Detect if user is speaking — interrupt agent if so
                if self.call_ctx.is_agent_speaking:
                    pcm = bytes(frame.data)
                    if not self.stt.is_silence(pcm):
                        await self._interrupt_agent()
                yield bytes(frame.data)

        async for transcription in self.stt.stream_transcriptions(pcm_generator()):
            if transcription.is_final and transcription.text.strip():
                log.info(f"[User]: {transcription.text}")
                await self._on_user_utterance(transcription)

    async def _on_user_utterance(self, transcription: Transcription):
        """
        Called each time the user finishes a complete utterance.
        Runs the LLM and speaks the response.
        """
        text = transcription.text.strip()
        if not text:
            return

        # Update context
        self.call_ctx.add_user_turn(text)
        self.call_ctx.is_user_speaking = False

        # Stream LLM response → pipe into TTS → publish audio
        self._tts_task = asyncio.create_task(
            self._generate_and_speak(text)
        )
        try:
            await self._tts_task
        except asyncio.CancelledError:
            log.info(f"[{self.call_ctx.contact_name}] Interrupted by user")

    async def _generate_and_speak(self, user_text: str):
        """
        Core pipeline: LLM streaming → sentence buffering → TTS → audio out.

        We buffer LLM tokens into complete sentences before sending to TTS.
        This gives TTS enough context to generate natural-sounding speech
        while keeping latency low (first sentence spoken within ~1s).
        """
        self.call_ctx.is_agent_speaking = True
        self._interrupted.clear()

        full_response = []
        sentence_buffer = ""
        sentence_enders = {".", "!", "?", "..."}

        try:
            async for token in self.llm.stream_response(
                persona_prompt=self.call_ctx.persona_prompt,
                history=self.call_ctx.get_llm_history(),
                user_message=user_text,
                memory_context=self.call_ctx.memory_context,
                turn_count=self.call_ctx.turn_count,
            ):
                if self._interrupted.is_set():
                    break

                sentence_buffer += token
                full_response.append(token)

                # Flush sentence to TTS as soon as it ends
                if any(sentence_buffer.rstrip().endswith(e) for e in sentence_enders):
                    sentence = sentence_buffer.strip()
                    if sentence:
                        log.info(f"[{self.call_ctx.contact_name}] Speaking: {sentence}")
                        await self._speak_text(sentence)
                    sentence_buffer = ""

            # Flush any remaining text
            if sentence_buffer.strip() and not self._interrupted.is_set():
                await self._speak_text(sentence_buffer.strip())

        finally:
            self.call_ctx.is_agent_speaking = False
            if full_response:
                response_text = "".join(full_response)
                self.call_ctx.add_agent_turn(response_text)
                log.info(f"[{self.call_ctx.contact_name}]: {response_text}")

    async def _speak_text(self, text: str):
        """Convert text to audio and publish to the LiveKit room."""
        if not text or self._interrupted.is_set():
            return

        voice_id = self.call_ctx.persona_prompt  # overridden in main.py with actual voice
        audio_bytes = await self.tts.synthesise(text, voice_id="af_heart")

        if self._interrupted.is_set():
            return

        await self._publish_audio(audio_bytes)

    async def _publish_audio(self, audio_bytes: bytes):
        """
        Push raw PCM/MP3 bytes into the LiveKit audio source frame by frame.
        LiveKit will mix and deliver this to all participants in the room.
        """
        # If MP3, decode to PCM first
        pcm_bytes = _decode_to_pcm(audio_bytes, SAMPLE_RATE)

        # Split into 60ms frames and publish each
        frame_bytes = SAMPLES_PER_FRAME * 2  # 16-bit = 2 bytes per sample
        for i in range(0, len(pcm_bytes), frame_bytes):
            if self._interrupted.is_set():
                return
            chunk = pcm_bytes[i:i + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk = chunk.ljust(frame_bytes, b"\x00")
            frame = rtc.AudioFrame(
                data=chunk,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=SAMPLES_PER_FRAME,
            )
            await self._audio_source.capture_frame(frame)
            # Small yield to prevent blocking the event loop
            await asyncio.sleep(0)

    async def _interrupt_agent(self):
        """
        User started speaking while agent is talking.
        Cancel current TTS stream immediately.
        """
        if self._tts_task and not self._tts_task.done():
            self._interrupted.set()
            self._tts_task.cancel()
            self.call_ctx.is_agent_speaking = False
            log.info("[Pipeline] Agent interrupted by user")

    async def _speak_greeting(self):
        """Speak a natural opening line when the call connects."""
        greeting = f"Hey, it's {self.call_ctx.contact_name}. How can I help you today?"
        log.info(f"[{self.call_ctx.contact_name}] Greeting user")
        self.call_ctx.add_agent_turn(greeting)
        await self._generate_and_speak.__func__(self, greeting) if False else None
        await self._speak_text(greeting)


def _decode_to_pcm(audio_bytes: bytes, target_sample_rate: int) -> bytes:
    """
    Decode MP3/audio bytes to raw 16-bit mono PCM at target sample rate.
    Uses PyAV (av library) for decoding.
    Falls back to returning as-is if already PCM.
    """
    try:
        import av
        import io
        import numpy as np

        container = av.open(io.BytesIO(audio_bytes))
        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=target_sample_rate,
        )
        pcm_chunks = []
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            for r in resampled:
                pcm_chunks.append(bytes(r.planes[0]))
        return b"".join(pcm_chunks)
    except Exception:
        # Already PCM or decode failed — return as-is
        return audio_bytes
