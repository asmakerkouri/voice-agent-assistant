"""The "brain" turns a transcript into a tool call and a short spoken answer.

- LLMBrain: an LLM with tool calling (Ollama, vLLM, OpenAI... via the OpenAI-compatible API)
- RuleBrain: a small offline intent + slot parser, no model needed; used in CI and as a fallback
  when the LLM is unreachable on site
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import httpx

from .text import words_to_numbers
from .tools import CHECKLISTS, TOOL_SCHEMAS, SiteTools


@dataclass
class Turn:
    transcript: str
    tool: str | None
    args: dict = field(default_factory=dict)
    reply: str = ""


DEFECTS = ["crack", "hollow", "bump", "leak", "stain", "corrosion", "rust", "spalling", "dent", "hole", "damp"]
SEVERITY = {"critical": "critical", "urgent": "critical", "dangerous": "critical", "major": "major",
            "serious": "major", "big": "major", "large": "major", "minor": "minor", "small": "minor",
            "slight": "minor"}
PLACES = r"(wall|level|floor|room|zone|column|beam|slab|ceiling|stair(?:case|s)?|corridor|block|bay|grid)"
OPS = {"plus": "+", "minus": "-", "times": "*", "multiplied by": "*", "divided by": "/", "x": "*", "over": "/"}
HELP = ("I can log a defect, list the defects, set a timer, tell the time, calculate, "
        "or read a checklist step.")


class RuleBrain:
    name = "rules"

    def parse(self, transcript: str) -> tuple[str | None, dict]:
        t = words_to_numbers(transcript.lower())
        t = re.sub(r"[^a-z0-9'.+\-*/ ]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()

        if re.search(r"\b(what time|time is it|current time|tell me the time)\b", t):
            return "get_time", {}
        if re.search(r"\b(timer|countdown|remind me in|alarm in)\b", t):
            m = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|minutes?|mins?|seconds?|secs?)", t)
            if m:
                k = {"h": 3600, "m": 60, "s": 1}[m.group(2)[0]]
                return "set_timer", {"seconds": int(float(m.group(1)) * k)}
        kind = next((d for d in DEFECTS if re.search(rf"\b{d}s?\b", t)), None)
        if not kind and re.search(r"\b(list|read|repeat|what|how many|show)\b.*\b(defects?|issues?|problems?)\b", t):
            return "list_defects", {}
        if kind:
            places = re.findall(rf"\b{PLACES}\s+([a-z]?\s?\d+[a-z]?|[a-z])\b", t)  # 'column b 4' -> B4
            location = ", ".join(f"{p} {v.replace(' ', '').upper()}" for p, v in places) or "unspecified"
            sev = next((v for k, v in SEVERITY.items() if re.search(rf"\b{k}\b", t)), "minor")
            return "report_defect", {"kind": "corrosion" if kind == "rust" else kind, "location": location,
                                     "severity": sev}
        proc = next((p for p in CHECKLISTS if p.split()[0] in t), None)
        if proc and re.search(r"\b(step|procedure|checklist|how do i|next)\b", t):
            m = re.search(r"\bstep (\d+)\b", t) or re.search(r"\b(\d+)(?:st|nd|rd|th)? step\b", t)
            return "checklist_step", {"procedure": proc, "step": int(m.group(1)) if m else 1}
        expr = t
        for w, op in OPS.items():
            expr = re.sub(rf"\b{w}\b", op, expr)
        m = re.search(r"(-?\d+(?:\.\d+)?(?:\s*[-+*/]\s*-?\d+(?:\.\d+)?)+)", expr)
        if m and re.search(r"\b(what is|what's|how much|calculate|compute|equals?)\b|[+*/]", expr):
            return "calculate", {"expression": re.sub(r"\s+", " ", m.group(1)).strip()}
        return None, {}

    def respond(self, transcript: str, tools: SiteTools) -> Turn:
        tool, args = self.parse(transcript)
        reply = tools.call(tool, args) if tool else f"Sorry, I did not get that. {HELP}"
        return Turn(transcript, tool, args, reply)


class LLMBrain:
    """One tool round: the LLM picks a tool, then turns its result into a short spoken sentence."""

    SYSTEM = ("You are a hands-free voice assistant for construction-site technicians. Use the tools. "
              "Answers are spoken aloud: one or two short sentences, no lists, no markdown.")

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self.name = model
        self.url = (base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.key = api_key or os.getenv("LLM_API_KEY", "ollama")
        self.http = httpx.Client(timeout=60)
        self.tools = [{"type": "function", "function": s} for s in TOOL_SCHEMAS]

    def _chat(self, messages: list[dict]) -> dict:
        r = self.http.post(f"{self.url}/chat/completions", headers={"Authorization": f"Bearer {self.key}"},
                           json={"model": self.name, "messages": messages, "tools": self.tools, "temperature": 0})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]

    def respond(self, transcript: str, tools: SiteTools) -> Turn:
        messages = [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": transcript}]
        msg = self._chat(messages)
        calls = msg.get("tool_calls") or []
        if not calls:
            return Turn(transcript, None, {}, msg.get("content") or HELP)
        c = calls[0]
        name, args = c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")
        result = tools.call(name, args)
        messages += [{"role": "assistant", "content": "", "tool_calls": [c]},
                     {"role": "tool", "tool_call_id": c["id"], "content": result}]
        final = self._chat(messages).get("content") or result
        return Turn(transcript, name, args, final)


class FallbackBrain:
    """LLM first; if it is unreachable (no network on site), the rule brain answers."""

    def __init__(self, primary, fallback=None):
        self.primary, self.fallback = primary, fallback or RuleBrain()
        self.name = f"{primary.name}+rules"

    def respond(self, transcript: str, tools: SiteTools) -> Turn:
        try:
            return self.primary.respond(transcript, tools)
        except httpx.HTTPError:
            return self.fallback.respond(transcript, tools)
