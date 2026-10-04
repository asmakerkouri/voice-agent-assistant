"""Speech I/O, fully offline on CPU (sherpa-onnx):
- STT: Whisper tiny.en (int8 ONNX)
- VAD: Silero, splits a recording into utterances
- TTS: Piper VITS voices
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import sherpa_onnx as so

MODELS = Path("models")
SR = 16000


def read_wav(path) -> tuple[np.ndarray, int]:
    """16-bit PCM wav from a path or a file-like object -> (float32 mono samples, sample rate)."""
    with wave.open(path if hasattr(path, "read") else str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return (x.reshape(-1, ch).mean(1) if ch > 1 else x), sr


def write_wav(path: str | Path, samples: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator, kind: str = "pink") -> np.ndarray:
    """Add noise at a given signal-to-noise ratio. Pink noise is closer to machinery than white noise."""
    n = rng.standard_normal(len(x))
    if kind == "pink":  # 1/f spectrum
        f = np.fft.rfft(n)
        f /= np.sqrt(np.maximum(np.arange(len(f)), 1))
        n = np.fft.irfft(f, len(x))
    p_sig = np.mean(x**2)
    n *= np.sqrt(p_sig / (10 ** (snr_db / 10)) / np.mean(n**2))
    return (x + n).astype(np.float32)


class STT:
    def __init__(self, model_dir: str | Path = MODELS / "sherpa-onnx-whisper-tiny.en", threads: int = 2):
        d = Path(model_dir)
        self.rec = so.OfflineRecognizer.from_whisper(
            encoder=str(d / "tiny.en-encoder.int8.onnx"), decoder=str(d / "tiny.en-decoder.int8.onnx"),
            tokens=str(d / "tiny.en-tokens.txt"), num_threads=threads, language="en", task="transcribe")

    def __call__(self, samples: np.ndarray, sr: int = SR) -> str:
        s = self.rec.create_stream()
        s.accept_waveform(sr, samples.astype(np.float32))
        self.rec.decode_stream(s)
        return s.result.text.strip()


class VAD:
    def __init__(self, model: str | Path = MODELS / "silero_vad.onnx"):
        self.cfg = so.VadModelConfig()
        self.cfg.silero_vad.model = str(model)
        self.cfg.sample_rate = SR

    def segments(self, samples: np.ndarray) -> list[tuple[float, np.ndarray]]:
        """(start time in s, samples) of each utterance of a 16 kHz recording."""
        vad = so.VoiceActivityDetector(self.cfg, buffer_size_in_seconds=120)
        out = []
        for i in range(0, len(samples), 512):
            vad.accept_waveform(samples[i:i + 512].astype(np.float32))
            while not vad.empty():
                out.append((vad.front.start / SR, np.array(vad.front.samples, dtype=np.float32)))
                vad.pop()
        vad.flush()
        while not vad.empty():
            out.append((vad.front.start / SR, np.array(vad.front.samples, dtype=np.float32)))
            vad.pop()
        return out


class TTS:
    def __init__(self, voice: str = "en_US-amy-low", models: str | Path = MODELS, threads: int = 2):
        d = Path(models) / f"vits-piper-{voice}"
        self.voice = voice
        self.tts = so.OfflineTts(so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(
            vits=so.OfflineTtsVitsModelConfig(model=str(d / f"{voice}.onnx"), tokens=str(d / "tokens.txt"),
                                              data_dir=str(d / "espeak-ng-data")), num_threads=threads)))

    def __call__(self, text: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        a = self.tts.generate(text, sid=0, speed=speed)
        return np.array(a.samples, dtype=np.float32), a.sample_rate
