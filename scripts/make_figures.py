"""README figures from eval/results/voice_eval.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
LEVELS = ["clean", "20 dB", "10 dB", "5 dB"]


def robustness(r: dict) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=120)
    for split, color, label in (("dev", "#9DB4CC", "dev set (30 commands)"),
                                ("test", "#1E90FF", "held-out test set (24 commands)")):
        w = [100 * r["speech"][f"{split}|{lv}"]["wer"] for lv in LEVELS]
        a = [100 * r["speech"][f"{split}|{lv}"]["intent_accuracy"] for lv in LEVELS]
        a1.plot(LEVELS, w, marker="o", color=color, lw=2, label=label)
        a2.plot(LEVELS, a, marker="o", color=color, lw=2, label=label)
        for x, (wv, av) in enumerate(zip(w, a, strict=True)):
            dy = 7 if split == "dev" else -14
            a1.annotate(f"{wv:.1f}", (x, wv), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=8,
                        color=color)
            a2.annotate(f"{av:.0f}", (x, av), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    a1.set_title("Speech recognition: word error rate (%)", loc="left", fontsize=10.5)
    a2.set_title("End to end: right tool and right arguments (%)", loc="left", fontsize=10.5)
    a2.set_ylim(0, 105)
    for ax in (a1, a2):
        ax.set_xlabel("pink noise added (signal-to-noise ratio)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    a1.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "robustness.png")
    plt.close(fig)


def latency(r: dict) -> None:
    lat = r["latency_ms"]
    stages = [("Speech-to-text\nWhisper tiny.en", lat["stt_ms"], "#1E90FF"),
              (f"Agent (rule brain)\nintent + tool: {lat['agent_ms']:.1f} ms", lat["agent_ms"], "#8250DF"),
              ("Text-to-speech\nPiper", lat["tts_ms"], "#2EA44F")]
    fig, ax = plt.subplots(figsize=(10, 2.4), dpi=120)
    left = 0
    for name, ms, c in stages:
        ax.barh(0, ms, left=left, color=c, height=0.5)
        if ms > 60:
            ax.text(left + ms / 2, 0, f"{ms:.0f} ms", ha="center", va="center", color="white", fontweight="bold")
        ax.text(left + ms / 2, -0.42, name, ha="center", va="top", fontsize=8.5)
        left += ms
    ax.set_xlim(0, left * 1.02)
    ax.set_ylim(-1.1, 0.5)
    ax.axis("off")
    ax.set_title(f"Latency of one voice turn on 2 CPU cores: {lat['total_ms']:.0f} ms on average "
                 f"(p95 {lat['p95_total_ms']:.0f} ms) for {lat['audio_in_s']:.1f} s of speech in, "
                 f"{lat['audio_out_s']:.1f} s out", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "latency.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads((ROOT / "eval" / "results" / "voice_eval.json").read_text())
    robustness(res)
    latency(res)
    print("figures written to", OUT)
