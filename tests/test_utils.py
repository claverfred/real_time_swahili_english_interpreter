import json
import subprocess
import sys

import pytest

from src.utils.logging import redact
from src.utils.manifest import FallbackLog, build_manifest, new_run_dir, write_manifest


def test_redaction_of_tokens(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "supersecretvalue123")
    out = redact("login with supersecretvalue123 and hf_abcdefghijklmnopqrstuvwxyz api_key=XYZ")
    assert "supersecretvalue123" not in out
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in out
    assert "XYZ" not in out and "api_key=***" in out


def test_run_dirs_are_unique_and_manifest_never_overwritten(tmp_path):
    d1, d2 = new_run_dir(tmp_path), new_run_dir(tmp_path)
    assert d1 != d2
    fb = FallbackLog()
    fb.add("device", "cuda", "cpu", "no GPU")
    m = build_manifest(run_id=d1.name, mode="test", config={"profile": "cpu"}, runtime={}, fallbacks=fb)
    path = write_manifest(d1, m)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["fallbacks"][0]["selected"] == "cpu"
    assert "packages" in data and "python" in data
    with pytest.raises(FileExistsError):
        write_manifest(d1, m)


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "src.cli", *args], capture_output=True, text=True)


def test_cli_diagnose_json():
    res = _cli("diagnose", "--json")
    report = json.loads(res.stdout)
    assert {"platform", "checks", "summary", "cuda"} <= report.keys()
    assert res.returncode in (0, 1)


def test_cli_config_show_resolved():
    res = _cli("config", "show", "--profile", "cpu", "--resolve", "--json")
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["runtime"]["device"] == "cpu"


def test_cli_bad_override_exits_2():
    res = _cli("config", "validate", "--profile", "cpu", "--set", "asr.beam_size=0")
    assert res.returncode == 2 and "error" in res.stderr
