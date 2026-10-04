"""FastAPI service.

POST /voice  (multipart .wav, 16 kHz mono) -> transcript, tool call, reply text, reply audio (base64 wav), timings
POST /text   {"text": "..."}               -> same, without speech recognition
"""

from __future__ import annotations

import base64
import io
import os
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .brain import FallbackBrain, LLMBrain, RuleBrain
from .pipeline import VoiceAssistant
from .speech import read_wav, wav_bytes

app = FastAPI(title="Voice agent for site technicians", version="1.0.0")


@lru_cache
def assistant() -> VoiceAssistant:
    model = os.getenv("LLM_MODEL")
    brain = FallbackBrain(LLMBrain(model)) if model else RuleBrain()
    return VoiceAssistant(brain=brain)


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/voice")
async def voice(file: UploadFile = File(...)) -> dict:
    try:
        samples, sr = read_wav(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(400, f"expected a PCM .wav file: {e}") from e
    vt = assistant().handle(samples, sr)
    return vt.to_dict() | {"audio_wav_base64": base64.b64encode(wav_bytes(vt.audio, vt.sample_rate)).decode()}


@app.post("/text")
def text(body: TextIn) -> dict:
    a = assistant()
    turn = a.brain.respond(body.text, a.tools)
    return {"tool": turn.tool, "args": turn.args, "reply": turn.reply}
