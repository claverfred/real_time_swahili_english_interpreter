"""Full pipeline with the real cpu-profile models. Slow; skipped unless models and the synthetic
Swahili clip are already on disk (no downloads during tests)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.mt.convert import find_converted

pytestmark = pytest.mark.slow
CLIP = Path("data/synthetic/sw_tts_v1/s03_sw-TZ-DaudiNeural.wav")


def test_interpret_cli_batch_export(tmp_path):
    if not CLIP.is_file():
        pytest.skip("synthetic Swahili clip not present")
    if find_converted(Path("cache"), "facebook/nllb-200-distilled-600M", "int8") is None:
        pytest.skip("converted NLLB not cached")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "SWI__STORAGE__OUTPUT_DIR": str(tmp_path)}
    res = subprocess.run([sys.executable, "-m", "src.cli", "interpret", "--input", str(CLIP), "--profile", "cpu",
                          "--export", "--quiet"], capture_output=True, text=True, env=env, timeout=600)
    if res.returncode != 0 and "not cached" in res.stderr.lower():
        pytest.skip(res.stderr[-300:])
    assert res.returncode == 0, res.stderr[-2000:]
    assert "EN [  0]" in res.stdout
    run = next(tmp_path.glob("interpret-*"))
    units = [json.loads(line) for line in (run / "units.jsonl").read_text(encoding="utf-8").splitlines()]
    assert units and all(u["english"] for u in units)
    assert all(u["lat_first_audio_ms"] is not None for u in units)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "interpret-batch" and manifest["extra"]["summary"]["counters"]["errors"] == 0
    assert (run / "speech_en.wav").stat().st_size > 10000
