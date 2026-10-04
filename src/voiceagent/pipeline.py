"""Voice pipeline: audio -> VAD -> speech-to-text -> agent (tools) -> text-to-speech -> audio, timed per stage."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .brain import RuleBrain, Turn
from .speech import SR, STT, TTS, VAD
from .tools import SiteTools


@dataclass
class Timings:
    stt_ms: float = 0
    agent_ms: float = 0
    tts_ms: float = 0
    audio_in_s: float = 0
    audio_out_s: float = 0

    @property
    def total_ms(self) -> float:
        return self.stt_ms + self.agent_ms + self.tts_ms


@dataclass
class VoiceTurn:
    turn: Turn
    audio: np.ndarray
    sample_rate: int
    timings: Timings = field(default_factory=Timings)

    def to_dict(self) -> dict:
        return {"transcript": self.turn.transcript, "tool": self.turn.tool, "args": self.turn.args,
                "reply": self.turn.reply, "timings": asdict(self.timings) | {"total_ms": self.timings.total_ms}}


class VoiceAssistant:
    def __init__(self, brain=None, stt: STT | None = None, tts: TTS | None = None, tools: SiteTools | None = None):
        self.brain = brain or RuleBrain()
        self.stt = stt or STT()
        self.tts = tts or TTS()
        self.tools = tools or SiteTools()
        self._vad: VAD | None = None

    def handle(self, samples: np.ndarray, sr: int = SR) -> VoiceTurn:
        """One spoken request -> one spoken answer."""
        t = Timings(audio_in_s=len(samples) / sr)
        t0 = time.perf_counter()
        text = self.stt(samples, sr)
        t1 = time.perf_counter()
        turn = self.brain.respond(text, self.tools)
        t2 = time.perf_counter()
        audio, out_sr = self.tts(turn.reply)
        t3 = time.perf_counter()
        t.stt_ms, t.agent_ms, t.tts_ms = 1000 * (t1 - t0), 1000 * (t2 - t1), 1000 * (t3 - t2)
        t.audio_out_s = len(audio) / out_sr
        return VoiceTurn(turn, audio, out_sr, t)

    def handle_recording(self, samples: np.ndarray, sr: int = SR) -> list[VoiceTurn]:
        """A long recording with several requests: split by voice activity, answer each one."""
        if sr != SR:
            raise ValueError("recordings must be 16 kHz mono")
        self._vad = self._vad or VAD()
        return [self.handle(seg, sr) for _, seg in self._vad.segments(samples)]
