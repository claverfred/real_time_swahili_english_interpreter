"""Render the README result charts as static SVG (no plotting dependencies).

    python scripts/make_readme_figures.py          # writes docs/images/*.svg

The numbers are copied from the measured results reported in README.md; update both together.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]    # categorical slots 1-4, fixed order
NEUTRAL_BAR = "#c3c2b7"
FONT = "font-family='Segoe UI, system-ui, -apple-system, Helvetica, Arial, sans-serif'"


def _svg(w: int, h: int, body: list[str], title: str) -> str:
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' "
            f"role='img' aria-label='{escape(title)}' {FONT}>\n<title>{escape(title)}</title>\n"
            f"<rect width='{w}' height='{h}' rx='8' fill='{SURFACE}' stroke='{GRID}'/>\n" + "\n".join(body) + "\n</svg>\n")


def _text(x, y, s, size=13, fill=INK, anchor="start", weight="normal") -> str:
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' "
            f"font-weight='{weight}'>{escape(s)}</text>")


def hbar_chart(name: str, title: str, subtitle: str, rows: list[tuple[str, float, bool]], xmax: float,
               ticks: list[float], fmt, note: str) -> None:
    """Single-series horizontal bars; highlighted rows use series 1, the rest a neutral."""
    label_w, left, right, top, row_h, bar_h = 330, 24, 70, 78, 34, 18
    w = 860
    plot_w = w - left - label_w - right
    h = top + row_h * len(rows) + 58
    x0 = left + label_w
    body = [_text(left, 30, title, 17, INK, weight="600"), _text(left, 52, subtitle, 12.5, INK2)]
    for t in ticks:                                                            # recessive grid
        x = x0 + plot_w * t / xmax
        body.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{top - 6}' y2='{top + row_h * len(rows)}' "
                    f"stroke='{GRID}' stroke-width='1'/>")
        body.append(_text(x, top + row_h * len(rows) + 18, fmt(t), 11.5, MUTED, "middle"))
    for i, (label, value, hi) in enumerate(rows):
        y = top + i * row_h
        bw = max(2.0, plot_w * value / xmax)
        body.append(_text(x0 - 12, y + bar_h - 4, label, 13, INK if hi else INK2, "end", "600" if hi else "normal"))
        body.append(f"<rect x='{x0}' y='{y}' width='{bw:.1f}' height='{bar_h}' rx='4' "
                    f"fill='{SERIES[0] if hi else NEUTRAL_BAR}'/>")
        body.append(_text(x0 + bw + 8, y + bar_h - 4, fmt(value), 12.5, INK, weight="600" if hi else "normal"))
    body.append(f"<line x1='{x0}' x2='{x0}' y1='{top - 6}' y2='{top + row_h * len(rows)}' stroke='{AXIS}'/>")
    body.append(_text(left, h - 14, note, 11.5, MUTED))
    (OUT / name).write_text(_svg(w, h, body, title), encoding="utf-8")


def stacked_breakdown(name: str) -> None:
    """Where the time goes between the speaker stopping and the first English audio."""
    parts = [("Waiting for silence (VAD)", 0.5), ("Swahili recognition (ASR)", 1.9),
             ("Translation (MT)", 0.5), ("First English audio (TTS)", 0.2)]
    total = sum(v for _, v in parts)
    w, left, right = 860, 24, 24
    plot_w, y, bar_h = w - left - right, 92, 34
    title = "Where the 3.1 s goes: speaker stops → first English audio"
    body = [_text(left, 30, title, 17, INK, weight="600"),
            _text(left, 52, "cpu profile, Phase 6 live simulation, ASR beam 1, partials off (median per stage)", 12.5, INK2)]
    x = float(left)
    for i, (_label, v) in enumerate(parts):
        seg = plot_w * v / total
        gap = 2 if i < len(parts) - 1 else 0                                   # 2px surface gap between fills
        body.append(f"<rect x='{x:.1f}' y='{y}' width='{seg - gap:.1f}' height='{bar_h}' rx='4' fill='{SERIES[i]}'/>")
        if seg > 60:
            body.append(_text(x + (seg - gap) / 2, y + 22, f"{v:.1f} s", 13,
                              INK if SERIES[i] == "#eda100" else "#ffffff", "middle", "600"))
        x += seg
    for t in (0, 1, 2, 3):
        tx = left + plot_w * t / total
        body.append(_text(tx, y + bar_h + 18, f"{t} s", 11.5, MUTED, "middle"))
    ly = y + bar_h + 48                                                        # legend (identity not by color alone)
    for i, (label, v) in enumerate(parts):
        lx, ry = left + (i % 2) * 400, ly + (i // 2) * 22
        body.append(f"<rect x='{lx}' y='{ry - 10}' width='12' height='12' rx='3' fill='{SERIES[i]}'/>")
        body.append(_text(lx + 18, ry, f"{label}  {v:.1f} s", 12.5, INK2))
    ly += 22
    body.append(_text(left, ly + 30,"Speech recognition is ~60% of the delay; with beam 5 (current default) it is ~0.3 s longer.",
                      11.5, MUTED))
    (OUT / name).write_text(_svg(w, ly + 46, body, title), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hbar_chart(
        "asr_wer_fleurs.svg",
        "Swahili speech recognition: word error rate (lower is better)",
        "FLEURS Swahili test, 40 real utterances, normalized WER, int8 CTranslate2 on CPU. Blue = the model in use",
        [("Whisper small (OpenAI)", 0.791, False),
         ("ElizabethMwangi whisper-small-swahili", 0.360, False),
         ("cdli whisper-small-swahili-afrivoice_cv", 0.354, False),
         ("Whisper large-v3-turbo (OpenAI), 3.5× slower", 0.349, False),
         ("Mollel/ASR-Swahili-Small, beam 1", 0.297, True),
         ("adoamesh whisper-small-swh (trained on FLEURS)", 0.290, False),
         ("Mollel/ASR-Swahili-Small, beam 5 (cpu default)", 0.275, True)],
        0.8, [0, 0.2, 0.4, 0.6, 0.8], lambda v: f"{v:.2f}" if v else "0",
        "adoamesh was fine-tuned on FLEURS itself, so its score on this test set is optimistic.")
    hbar_chart(
        "latency_first_audio.svg",
        "Delay from the speaker stopping to the first English audio",
        "Median per sentence, cpu profile, 10-core CPU, no GPU. Blue = current default settings (ASR beam 5)",
        [("CLI live-sim, ASR partials on (beam 1)", 4.1, False),
         ("CLI live-sim, partials off (beam 1)", 3.1, False),
         ("Live laptop mic, turn-taking, session 1 (beam 1)", 3.7, False),
         ("Live laptop mic, turn-taking, session 2 (beam 1)", 3.9, False),
         ("Server WebSocket client, 30 s clip (beam 5)", 2.7, True),
         ("Browser UI, headless Edge, 30 s clip (beam 5)", 3.3, True)],
        5.0, [0, 1, 2, 3, 4, 5], lambda v: f"{v:.1f} s" if v else "0",
        "Live mic: real speech, 14 + 11 sentences. 30 s clip: synthetic speech, 6 sentences. Server run reused cached translations.")
    stacked_breakdown("latency_breakdown.svg")
    for p in sorted(OUT.glob("*.svg")):
        print("wrote", p)


if __name__ == "__main__":
    main()
