"""Benchmark of the voice agent, fully offline.

Spoken commands are synthesised with 3 voices that are NOT the assistant's voice (US male, UK male,
US female), mixed with pink noise at several signal-to-noise ratios, then go through the real
pipeline. Measured: word error rate of speech recognition, end-to-end intent accuracy (right tool
and right arguments), latency of each stage.

    PYTHONPATH=src python eval/run_eval.py      -> eval/results/voice_eval.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceagent.brain import RuleBrain  # noqa: E402
from voiceagent.pipeline import VoiceAssistant  # noqa: E402
from voiceagent.speech import STT, TTS, add_noise, read_wav  # noqa: E402
from voiceagent.text import wer  # noqa: E402
from voiceagent.tools import safe_eval  # noqa: E402

VOICES = {"en_US-ryan-low": "US male", "en_GB-alan-low": "UK male", "en_US-lessac-low": "US female"}
SNRS = [None, 20, 10, 5]  # None = clean


def load(split: str) -> list[dict]:
    return [json.loads(x) for x in (ROOT / "eval" / f"{split}_commands.jsonl").read_text().splitlines() if x.strip()]


def correct(cmd: dict, tool: str | None, args: dict) -> bool:
    if tool != cmd["tool"]:
        return False
    exp = cmd["args"]
    if tool == "calculate":
        try:
            return abs(safe_eval(args["expression"]) - exp["value"]) < 1e-6
        except Exception:
            return False
    return all(args.get(k) == v for k, v in exp.items())


def main() -> dict:
    stt, brain, rng = STT(), RuleBrain(), np.random.default_rng(0)
    splits = {s: load(s) for s in ("dev", "test")}
    res: dict = {"text_only": {}, "speech": {}, "voices": VOICES}

    # 1. upper bound: the brain on the exact text
    for s, cmds in splits.items():
        ok = [correct(c, *brain.parse(c["text"])) for c in cmds]
        res["text_only"][s] = {"correct": sum(ok), "n": len(ok)}

    # 2. speech: 3 voices x 4 noise levels
    errors = []
    for voice in VOICES:
        tts = TTS(voice)
        for s, cmds in splits.items():
            for c in cmds:
                clean, sr = tts(c["text"])
                for snr in SNRS:
                    audio = clean if snr is None else add_noise(clean, snr, rng)
                    t0 = time.perf_counter()
                    hyp = stt(audio, sr)
                    stt_ms = 1000 * (time.perf_counter() - t0)
                    e, n = wer(c["text"], hyp)
                    ok = correct(c, *brain.parse(hyp))
                    key = f"{s}|{'clean' if snr is None else f'{snr} dB'}"
                    r = res["speech"].setdefault(key, defaultdict(float))
                    r["edits"] += e
                    r["words"] += n
                    r["correct"] += ok
                    r["n"] += 1
                    r["stt_ms"] += stt_ms
                    r["audio_s"] += len(audio) / sr
                    if not ok and snr in (None, 20):
                        errors.append({"split": s, "voice": voice, "snr": snr, "text": c["text"], "heard": hyp})
    for r in res["speech"].values():
        r["wer"] = round(r["edits"] / r["words"], 4)
        r["intent_accuracy"] = round(r["correct"] / r["n"], 4)
        r["stt_rtf"] = round(r["stt_ms"] / 1000 / r["audio_s"], 3)
    res["errors_clean_or_20db"] = errors

    # 3. sanity check of the STT on real human speech (LibriSpeech samples shipped with the model)
    d = ROOT / "models" / "sherpa-onnx-whisper-tiny.en" / "test_wavs"
    refs = dict(line.split(" ", 1) for line in (d / "trans.txt").read_text().splitlines() if line.strip())
    e = n = 0
    for name, ref in refs.items():
        x, sr = read_wav(d / name)
        de, dn = wer(ref, stt(x, sr))
        e, n = e + de, n + dn
    res["librispeech_samples"] = {"files": len(refs), "words": n, "wer": round(e / n, 4)}

    # 4. end-to-end latency of a full turn (speech in -> speech out), assistant voice
    va = VoiceAssistant(brain=brain, stt=stt)
    tts = TTS("en_US-ryan-low")
    lat = []
    for c in splits["test"]:
        audio, sr = tts(c["text"])
        lat.append(va.handle(audio, sr).timings)
    res["latency_ms"] = {k: round(float(np.mean([getattr(t, k) for t in lat])), 1)
                         for k in ("stt_ms", "agent_ms", "tts_ms", "total_ms", "audio_in_s", "audio_out_s")}
    res["latency_ms"]["p95_total_ms"] = round(float(np.percentile([t.total_ms for t in lat], 95)), 1)

    out = ROOT / "eval" / "results" / "voice_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "errors_clean_or_20db"}, indent=2))
    print(f"{len(errors)} errors at clean/20 dB:")
    for x in errors:
        print(" ", x)
    return res


if __name__ == "__main__":
    main()
