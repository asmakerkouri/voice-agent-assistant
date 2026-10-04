"""Talk to the assistant with a .wav file (16 kHz mono) or with text.

    python -m voiceagent.cli --wav samples/request.wav --out output/answer.wav
    python -m voiceagent.cli --say "There is a crack on wall twelve, level two"   # synthesise the request first
    python -m voiceagent.cli --wav samples/session.wav --session                   # several requests (VAD)
    python -m voiceagent.cli --say "..." --llm qwen2.5:7b                           # LLM brain via Ollama
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .brain import FallbackBrain, LLMBrain, RuleBrain
from .pipeline import VoiceAssistant
from .speech import TTS, read_wav, write_wav


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav")
    src.add_argument("--say", help="text spoken by a test voice, then processed as audio")
    p.add_argument("--session", action="store_true", help="split a long recording into requests")
    p.add_argument("--llm", help="model on an OpenAI-compatible endpoint (default: offline rule brain)")
    p.add_argument("--out", default="output/answer.wav")
    a = p.parse_args()

    brain = FallbackBrain(LLMBrain(a.llm)) if a.llm else RuleBrain()
    va = VoiceAssistant(brain=brain)
    if a.say:
        samples, sr = TTS("en_US-ryan-low")(a.say)
        write_wav(Path(a.out).with_name("request.wav"), samples, sr)
    else:
        samples, sr = read_wav(a.wav)
    turns = va.handle_recording(samples, sr) if a.session else [va.handle(samples, sr)]
    for i, vt in enumerate(turns):
        print(json.dumps(vt.to_dict(), indent=2, ensure_ascii=False))
        out = Path(a.out) if len(turns) == 1 else Path(a.out).with_stem(f"{Path(a.out).stem}_{i + 1}")
        print("answer audio:", write_wav(out, vt.audio, vt.sample_rate))


if __name__ == "__main__":
    main()
