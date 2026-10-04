"""Tools of the hands-free assistant for site technicians, with their JSON schemas."""

from __future__ import annotations

import ast
import csv
import datetime as dt
import operator
from dataclasses import dataclass, field
from pathlib import Path

CHECKLISTS = {
    "scanner calibration": [
        "Place the calibration target two metres in front of the scanner.",
        "Check that the target is fully visible and evenly lit.",
        "Start the calibration from the scanner menu and wait for the beep.",
        "Check that the reported error is below one millimetre, then save.",
    ],
    "robot start-up": [
        "Check that the emergency stop is released and the area is clear.",
        "Switch on the controller, then the arm.",
        "Run the homing sequence and wait until the status light is green.",
        "Load the inspection program and confirm the zone on the tablet.",
    ],
}

TOOL_SCHEMAS = [
    {"name": "report_defect", "description": "Log a defect found on site.",
     "parameters": {"type": "object", "required": ["kind", "location"], "properties": {
         "kind": {"type": "string", "description": "crack, hollow, bump, leak, stain, corrosion..."},
         "location": {"type": "string", "description": "e.g. 'wall 12, level 2'"},
         "severity": {"type": "string", "enum": ["minor", "major", "critical"]}}}},
    {"name": "list_defects", "description": "Read back the defects logged so far.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "set_timer", "description": "Start a countdown.",
     "parameters": {"type": "object", "required": ["seconds"], "properties": {"seconds": {"type": "integer"}}}},
    {"name": "get_time", "description": "Current time.", "parameters": {"type": "object", "properties": {}}},
    {"name": "calculate", "description": "Evaluate an arithmetic expression such as '12 * 7'.",
     "parameters": {"type": "object", "required": ["expression"], "properties": {"expression": {"type": "string"}}}},
    {"name": "checklist_step", "description": "Read one step of a procedure (scanner calibration, robot start-up).",
     "parameters": {"type": "object", "required": ["procedure"], "properties": {
         "procedure": {"type": "string"}, "step": {"type": "integer", "description": "1-based, default 1"}}}},
]

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg}


def safe_eval(expr: str) -> float:
    """Arithmetic only: no names, no calls, no attribute access."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("only arithmetic is allowed")
    return ev(ast.parse(expr, mode="eval").body)


@dataclass
class SiteTools:
    log_path: Path | None = None
    defects: list[dict] = field(default_factory=list)
    timers: list[int] = field(default_factory=list)

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, name, None)
        if fn is None or name not in {s["name"] for s in TOOL_SCHEMAS}:
            return f"Unknown tool {name}."
        try:
            return fn(**args)
        except Exception as e:
            return f"Error: {e}"

    def report_defect(self, kind: str, location: str, severity: str = "minor") -> str:
        row = {"id": len(self.defects) + 1, "time": dt.datetime.now().isoformat(timespec="seconds"),
               "kind": kind, "location": location, "severity": severity}
        self.defects.append(row)
        if self.log_path:
            new = not self.log_path.exists()
            with self.log_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
        return f"Defect {row['id']} logged: {severity} {kind}, {location}."

    def list_defects(self) -> str:
        if not self.defects:
            return "No defect logged yet."
        items = "; ".join(f"{d['id']}, {d['severity']} {d['kind']} on {d['location']}" for d in self.defects)
        return f"{len(self.defects)} defect{'s' if len(self.defects) > 1 else ''}: {items}."

    def set_timer(self, seconds: int) -> str:
        self.timers.append(int(seconds))
        m, s = divmod(int(seconds), 60)
        human = " and ".join(p for p in (f"{m} minute{'s' * (m > 1)}" if m else "",
                                         f"{s} second{'s' * (s > 1)}" if s else "") if p)
        return f"Timer set for {human}."

    def get_time(self) -> str:
        return f"It is {dt.datetime.now().strftime('%H:%M')}."

    def calculate(self, expression: str) -> str:
        v = safe_eval(expression)
        v = int(v) if float(v).is_integer() else round(v, 3)
        return f"{expression.replace('*', 'times').replace('/', 'divided by')} is {v}."

    def checklist_step(self, procedure: str, step: int = 1) -> str:
        key = next((k for k in CHECKLISTS if k.split()[0] in procedure.lower()), None)
        if key is None:
            return f"I don't know the procedure '{procedure}'. I know: {', '.join(CHECKLISTS)}."
        steps = CHECKLISTS[key]
        if not 1 <= step <= len(steps):
            return f"The {key} procedure has {len(steps)} steps."
        return f"{key.capitalize()}, step {step} of {len(steps)}: {steps[step - 1]}"
