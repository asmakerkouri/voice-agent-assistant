from pathlib import Path

import numpy as np
import pytest

from voiceagent.brain import FallbackBrain, LLMBrain, RuleBrain
from voiceagent.text import normalize, wer, words_to_numbers
from voiceagent.tools import SiteTools, safe_eval

MODELS = Path(__file__).resolve().parents[1] / "models"
needs_models = pytest.mark.skipif(not (MODELS / "silero_vad.onnx").exists(),
                                  reason="run scripts/download_models.sh first")


# ------------------------------------------------------------------ text
@pytest.mark.parametrize("text, expected", [
    ("set a timer for twenty five minutes", "set a timer for 25 minutes"),
    ("wall twelve, level two.", "wall 12, level 2."),
    ("one hundred and five", "105"),
    ("three point five times four", "3.5 times 4"),
    ("one and two", "1 and 2"),
])
def test_words_to_numbers(text, expected):
    assert words_to_numbers(text) == expected


def test_wer():
    assert wer("There is a crack on wall twelve", "there is a crack on wall 12.") == (0, 7)
    assert wer("set a timer", "set the timer") == (1, 3)
    assert normalize("Wall 12, Level 2!") == "wall 12 level 2"


# ----------------------------------------------------------------- brain
@pytest.mark.parametrize("text, tool, args", [
    ("There is a crack on wall 12, level 2. It looks major.", "report_defect",
     {"kind": "crack", "location": "wall 12, level 2", "severity": "major"}),
    ("I found a hollow on column B 4, it's critical.", "report_defect",
     {"kind": "hollow", "location": "column B4", "severity": "critical"}),
    ("Set a timer for 5 minutes.", "set_timer", {"seconds": 300}),
    ("What time is it?", "get_time", {}),
    ("How many defects did I report today?", "list_defects", {}),
    ("Read step 3 of the scanner calibration procedure.", "checklist_step",
     {"procedure": "scanner calibration", "step": 3}),
])
def test_rule_brain(text, tool, args):
    assert RuleBrain().parse(text) == (tool, args)


def test_rule_brain_calculates():
    tool, args = RuleBrain().parse("What is twelve times seven?")
    assert tool == "calculate" and safe_eval(args["expression"]) == 84


def test_unknown_request():
    turn = RuleBrain().respond("Play some music", SiteTools())
    assert turn.tool is None and "I can" in turn.reply


def test_fallback_when_llm_unreachable():
    brain = FallbackBrain(LLMBrain("any", base_url="http://127.0.0.1:9"))
    assert brain.respond("What time is it?", SiteTools()).tool == "get_time"


# ----------------------------------------------------------------- tools
def test_tools(tmp_path):
    t = SiteTools(log_path=tmp_path / "defects.csv")
    assert t.report_defect("crack", "wall 12", "major") == "Defect 1 logged: major crack, wall 12."
    assert "1 defect: 1, major crack on wall 12" in t.list_defects()
    assert (tmp_path / "defects.csv").read_text().count("\n") == 2
    assert t.set_timer(90) == "Timer set for 1 minute and 30 seconds."
    assert t.calculate("200 / 8") == "200 divided by 8 is 25."
    assert "step 2 of 4" in t.checklist_step("robot start-up", 2)


def test_calculator_refuses_code():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('ls')")
    assert "Error" in SiteTools().call("calculate", {"expression": "9 ** 9 ** 9"})


# --------------------------------------------- speech (needs the models)
@needs_models
def test_full_voice_turn():
    from voiceagent.pipeline import VoiceAssistant
    from voiceagent.speech import TTS

    va = VoiceAssistant()
    audio, sr = TTS("en_US-ryan-low")("Log a small leak in room three.")
    vt = va.handle(audio, sr)
    assert vt.turn.tool == "report_defect" and vt.turn.args["location"] == "room 3"
    assert vt.audio.size > sr and vt.timings.total_ms > 0


@needs_models
def test_vad_splits_a_session():
    from voiceagent.pipeline import VoiceAssistant
    from voiceagent.speech import TTS

    tts, va = TTS("en_GB-alan-low"), VoiceAssistant()
    a, sr = tts("Set a timer for two minutes.")
    b, _ = tts("What time is it?")
    silence = np.zeros(sr, dtype=np.float32)
    turns = va.handle_recording(np.concatenate([silence, a, silence, b, silence]), sr)
    assert [t.turn.tool for t in turns] == ["set_timer", "get_time"]
