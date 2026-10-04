"""Text normalisation for spoken language: number words, punctuation, word error rate."""

from __future__ import annotations

import re

UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}


def words_to_numbers(text: str) -> str:
    """'set a timer for twenty five minutes' -> 'set a timer for 25 minutes'"""
    out: list[str] = []
    num, last = 0, None  # last: None | unit | tens | hundred

    def flush(sep: str = " ") -> None:
        nonlocal num, last
        if last is not None:
            out.append(f"{num}{sep}")
        num, last = 0, None

    for tok in re.split(r"(\s+|-)", text):
        m = re.fullmatch(r"([A-Za-z]+)([.,!?;:]*)", tok)
        w, punct = (m.group(1).lower(), m.group(2)) if m else (tok.lower(), "")
        if w in UNITS or w in TENS or w == "hundred":
            if w == "hundred" and last in ("unit", None):
                num, last = max(num, 1) * 100, "hundred"
            elif w in TENS:
                if last in ("unit", "tens"):
                    flush()
                num, last = num + TENS[w], "tens"
            elif w in UNITS:
                if last == "unit" or (last == "tens" and UNITS[w] >= 10):
                    flush()
                num, last = num + UNITS[w], "unit"
            else:
                flush()
                out.append(tok)
            if punct:  # 'twelve,' ends the number
                flush(punct)
            continue
        if last is not None and (tok.isspace() or tok == "-" or (w == "and" and last == "hundred")):
            continue
        flush()
        out.append(tok)
    flush("")
    text = re.sub(r"\s+", " ", "".join(out)).strip()
    return re.sub(r"\b(\d+) point (\d+)\b", r"\1.\2", text)  # 'three point five' -> 3.5


def normalize(text: str) -> str:
    text = words_to_numbers(text.lower())
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wer(reference: str, hypothesis: str) -> tuple[int, int]:
    """(word edits, reference words) after normalisation; WER = edits / words."""
    r, h = normalize(reference).split(), normalize(hypothesis).split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)
