"""Write a finished interpreter session to a run directory (CLI ``--export`` and the server).

Files: ``transcript_sw.txt``, ``translation_en.txt`` (only with ``logging.log_text``),
``units.jsonl`` (text fields nulled without log_text), ``speech_en.wav``, ``events.jsonl``,
``events.csv`` and ``manifest.json``. Input (microphone or uploaded) audio is never written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.manifest import build_manifest, new_run_dir, write_manifest


def export_session(interp: Any, output_dir: Path, *, mode: str, speech: np.ndarray | None, speech_sr: int,
                   summary: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    from ..audio import save_wav

    cfg = interp.cfg
    run_dir = new_run_dir(Path(output_dir), "interpret")
    keep_text = cfg.logging.log_text
    units = interp.units()
    if keep_text:
        (run_dir / "transcript_sw.txt").write_text("\n".join(u["swahili"] for u in units) + "\n", encoding="utf-8")
        (run_dir / "translation_en.txt").write_text("\n".join(u["english"] for u in units) + "\n", encoding="utf-8")
    rows = [u if keep_text else {**u, "swahili": None, "english": None} for u in units]
    (run_dir / "units.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")
    if speech is not None and len(speech):
        save_wav(run_dir / "speech_en.wav", speech, speech_sr)
    interp.recorder.export_jsonl(run_dir / "events.jsonl")
    interp.recorder.export_csv(run_dir / "events.csv")
    write_manifest(run_dir, build_manifest(
        run_id=run_dir.name, mode=mode, config=cfg.model_dump(mode="json", by_alias=True),
        runtime=interp.runtime.to_dict(), fallbacks=interp.fallbacks,
        extra={**(extra or {}), "models": interp.model_status(), "summary": summary, "text_logged": keep_text}))
    return run_dir
