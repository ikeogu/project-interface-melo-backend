"""
STT plugin for LiveKit Agents — wraps self-hosted Faster-Whisper.

LiveKit Agents expects an STT plugin that:
  1. Accepts raw PCM audio frames
  2. Returns transcription text (with end-of-utterance detection)

We batch audio frames until silence is detected (VAD), then send
the chunk to Faster-Whisper for transcription.

Faster-Whisper server: https://github.com/fedirz/faster-whisper-server
Deploy:
  docker run -p 8001:8000 fedirz/faster-whisper-server:latest-cpu \
    -e WHISPER__MODEL=Systran/faster-whisper-small
"""
import httpx
import asyncio
import io
import numpy as np
from dataclasses import dataclass
from typing import AsyncGenerator

# Silence detection config
SILENCE_THRESHOLD = 0.01       # RMS amplitude below this = silence
SILENCE_DURATION_MS = 800      # ms of silence before we consider utterance done
SAMPLE_RATE = 16000            # Faster-Whisper expects 16kHz
BYTES_PER_SAMPLE = 2           # 16-bit PCM


@dataclass
class Transcription:
    text: str
    is_final: bool
    confidence: float = 1.0


class FasterWhisperSTT:
    """
    Self-hosted Faster-Whisper STT.
    Compatible with LiveKit Agents STT plugin interface.
    """

    def __init__(self, service_url: str, language: str = "en"):
        self.service_url = service_url.rstrip("/")
        self.language = language
        self._audio_buffer: list[bytes] = []
        self._silence_frames: int = 0
        self._frames_per_silence_check = int(SAMPLE_RATE * 0.1)  # 100ms chunks

    async def transcribe_segment(self, audio_bytes: bytes) -> str:
        """
        Send a complete audio segment to Faster-Whisper.
        audio_bytes: raw 16kHz 16-bit mono PCM.
        Returns transcript text.
        """
        if not audio_bytes:
            return ""

        # Convert raw PCM to WAV for the HTTP API
        wav_bytes = _pcm_to_wav(audio_bytes, SAMPLE_RATE)

        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                f"{self.service_url}/v1/audio/transcriptions",
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={
                    "model": "Systran/faster-whisper-small",
                    "language": self.language,
                    "response_format": "json",
                },
            )
            if r.status_code != 200:
                return ""
            return r.json().get("text", "").strip()

    def is_silence(self, pcm_bytes: bytes) -> bool:
        """Detect silence using RMS energy of the audio chunk."""
        if len(pcm_bytes) < 2:
            return True
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples ** 2)))
        return rms < SILENCE_THRESHOLD

    async def stream_transcriptions(
        self, audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[Transcription, None]:
        """
        Consume a stream of raw PCM chunks.
        Yields Transcription objects when end-of-utterance is detected.
        Used by the LiveKit pipeline to process mic input in real time.
        """
        buffer: list[bytes] = []
        silence_count = 0
        silence_frames_needed = int(SILENCE_DURATION_MS / 100)  # at 100ms per frame

        async for chunk in audio_stream:
            if self.is_silence(chunk):
                silence_count += 1
                if buffer and silence_count >= silence_frames_needed:
                    # End of utterance — transcribe what we have
                    audio_data = b"".join(buffer)
                    text = await self.transcribe_segment(audio_data)
                    if text:
                        yield Transcription(text=text, is_final=True)
                    buffer = []
                    silence_count = 0
            else:
                silence_count = 0
                buffer.append(chunk)

        # Flush any remaining audio at stream end
        if buffer:
            audio_data = b"".join(buffer)
            text = await self.transcribe_segment(audio_data)
            if text:
                yield Transcription(text=text, is_final=True)


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM bytes in a WAV container."""
    import struct
    num_samples = len(pcm_bytes) // BYTES_PER_SAMPLE
    byte_rate = sample_rate * BYTES_PER_SAMPLE
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm_bytes), b"WAVE",
        b"fmt ", 16, 1, 1,            # PCM, mono
        sample_rate, byte_rate,
        BYTES_PER_SAMPLE, 16,          # block align, bits per sample
        b"data", len(pcm_bytes),
    )
    return header + pcm_bytes
