"""
Minimal livekit-agents TTS plugin wrapping edge-tts (Microsoft, free, no limits).

Usage in agent_worker.py:
    from edge_tts_plugin import EdgeTTS
    tts = EdgeTTS(voice="en-US-AriaNeural")
"""
import asyncio
from dataclasses import dataclass

import edge_tts
from livekit.agents import tts, utils
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS


# Map contact voice_id → edge-tts voice name.
# Falls back to VOICE_MAP.get(voice_id, DEFAULT_VOICE).
VOICE_MAP: dict[str, str] = {
    # ElevenLabs IDs used in the DB — map to rough equivalents
    "EXAVITQu4vr4xnSDxMaL": "en-US-AriaNeural",    # Alex / Sarah → Aria (warm female)
    "ThT5KcBeYPX3keUQqHPh": "en-US-JennyNeural",   # Maya → Jenny (calm female)
    "VR6AewLTigWG4xSOukaG": "en-US-GuyNeural",     # Father James → Guy (deep male)
    "af_heart": "en-US-AriaNeural",                 # Kokoro default
}
DEFAULT_VOICE = "en-US-AriaNeural"
SAMPLE_RATE = 24000


class EdgeTTS(tts.TTS):
    def __init__(self, *, voice_id: str = DEFAULT_VOICE):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        self._voice = VOICE_MAP.get(voice_id, voice_id if voice_id.startswith("en-") else DEFAULT_VOICE)

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "_EdgeStream":
        return _EdgeStream(tts=self, input_text=text, conn_options=conn_options)

    def update_options(self, *, voice_id: str | None = None) -> None:
        if voice_id:
            self._voice = VOICE_MAP.get(voice_id, voice_id if voice_id.startswith("en-") else DEFAULT_VOICE)


class _EdgeStream(tts.ChunkedStream):
    def __init__(self, *, tts: EdgeTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: EdgeTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        communicate = edge_tts.Communicate(self._input_text, self._tts._voice)

        # Collect all MP3 chunks first (edge-tts streams small chunks)
        mp3_bytes = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_bytes.extend(chunk["data"])

        if not mp3_bytes:
            raise tts.TTSError("edge-tts returned no audio data")

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/mp3",
        )
        output_emitter.push(bytes(mp3_bytes))
        output_emitter.flush()
