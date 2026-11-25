"""Phase 9: Colab helpers, notebook session wrapper, run summaries and the generated notebooks."""

import ast
import asyncio
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

from src.config import load_config
from src.eval.summary import latency_row, mt_row, speech_rows, summarize_run
from src.mt import Translator
from src.pipeline import Components, InterpreterSession
from src.platform import colab
from src.tts import Synthesizer
from tests.test_asr_streaming import FakeAsr
from tests.test_interpreter import TWO_UTTERANCES, LevelVad
from tests.test_mt import FakeEngine
from tests.test_tts import FakeBackend

REPO = Path(__file__).resolve().parents[1]


def _can_symlink(tmp_path: Path) -> bool:
    try:
        (tmp_path / "probe").symlink_to(tmp_path, target_is_directory=True)
        return True
    except OSError:
        return False


# -- storage ---------------------------------------------------------------------------------------
def test_persistent_root_prefers_mounted_drive(tmp_path):
    drive = tmp_path / "drive"
    root, on_drive = colab.persistent_root("x", drive_mount=str(drive), local_base=str(tmp_path / "local"))
    assert (root, on_drive) == (tmp_path / "local" / "x", False)
    (drive / "MyDrive").mkdir(parents=True)
    root, on_drive = colab.persistent_root("x", drive_mount=str(drive), local_base=str(tmp_path / "local"))
    assert (root, on_drive) == (drive / "MyDrive" / "x", True)
    assert colab.persistent_root("x", use_drive=False, drive_mount=str(drive), local_base=str(tmp_path))[1] is False


def test_setup_storage_sets_env_and_feeds_the_colab_profile(tmp_path):
    env: dict[str, str] = {}
    repo = tmp_path / "repo"
    repo.mkdir()
    info = colab.setup_storage(repo, use_drive=False, environ=env, local_base=str(tmp_path))
    root = Path(info["root"])
    assert env[colab.ROOT_ENV] == str(root) and env["SWI_PROFILE"] == "colab"
    assert all((root / d).is_dir() for d in ("cache", "results", "env", "data/fleurs_sw"))
    assert info["notes"]                                    # warns that local disk is not persistent
    cfg = load_config("colab", environ=env)
    assert Path(cfg.storage.cache_dir) == root / "cache"
    assert Path(cfg.storage.output_dir) == root / "results"
    default = load_config("colab", environ={})
    assert default.storage.cache_dir == "/content/swahili_interpreter/cache"


def test_link_persistent_dirs_moves_files_and_keeps_clashes(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks unavailable (Windows without Developer Mode)")
    repo, root = tmp_path / "repo", tmp_path / "root"
    (repo / "data/a").mkdir(parents=True)
    (repo / "data/a/new.txt").write_text("from checkout")
    (repo / "data/a/same.txt").write_text("checkout copy")
    (root / "data/a").mkdir(parents=True)
    (root / "data/a/same.txt").write_text("persistent copy")
    done = colab.link_persistent_dirs(repo, root, ["data/a", "data/b"])
    assert len(done) == 2 and (repo / "data/a").is_symlink()
    assert (root / "data/a/new.txt").read_text() == "from checkout"
    assert (root / "data/a/same.txt").read_text() == "persistent copy"
    backups = list((repo / "data").glob("a.local-*"))
    assert len(backups) == 1 and (backups[0] / "same.txt").read_text() == "checkout copy"   # nothing lost
    assert colab.link_persistent_dirs(repo, root, ["data/a"]) == []                         # idempotent


# -- CUDA libraries, constraints, restart check ----------------------------------------------------
def test_cuda_library_dirs_are_prepended_once(tmp_path):
    for pkg in ("cudnn", "cublas"):
        (tmp_path / "nvidia" / pkg / "lib").mkdir(parents=True)
    env = {"LD_LIBRARY_PATH": "/usr/lib"}
    added = colab.export_cuda_library_path(env, [str(tmp_path)])
    assert len(added) == 2 and env["LD_LIBRARY_PATH"].endswith("/usr/lib")
    assert colab.export_cuda_library_path(env, [str(tmp_path)]) == []


def test_protected_constraints_pin_preinstalled_packages():
    pins = colab.protected_constraints(["numpy", "surely-not-installed-pkg"])
    assert len(pins) == 1 and pins[0].startswith("numpy==")


def test_restart_required_detects_version_drift():
    class Mod:
        __version__ = "0.0.1"
    assert colab.restart_required({"numpy": Mod()}) == [f"numpy: imported 0.0.1, installed {colab._version('numpy')}"]
    import numpy
    assert colab.restart_required({"numpy": numpy}) == []


def test_outside_colab_helpers_degrade_cleanly(tmp_path, monkeypatch):
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    if "google.colab" in sys.modules:
        pytest.skip("running inside Colab")
    assert colab.mount_drive() is False and colab.download(tmp_path) is False and colab.proxy_url(8000) is None
    with pytest.raises(RuntimeError):
        colab.record_browser_audio(tmp_path)
    with pytest.raises(RuntimeError):
        colab.upload_audio(tmp_path)


# -- export and CLI runner -------------------------------------------------------------------------
def test_zip_runs_keeps_run_folder_names(tmp_path):
    run = tmp_path / "eval-x"
    (run / "sub").mkdir(parents=True)
    (run / "results.json").write_text("{}")
    (run / "sub" / "a.wav").write_bytes(b"RIFF")
    extra = tmp_path / "table.csv"
    extra.write_text("a,b")
    info = colab.zip_runs([run], tmp_path / "out" / "all.zip", [extra, tmp_path / "missing.csv"])
    with zipfile.ZipFile(info["zip"]) as z:
        assert sorted(z.namelist()) == ["eval-x/results.json", "eval-x/sub/a.wav", "table.csv"]
    assert info["files"] == 3


def test_require_gpu_stops_gpu_profiles_on_a_cpu_runtime():
    from src.platform.cuda import CudaStatus, GpuInfo
    none = CudaStatus(False, "none")
    with pytest.raises(RuntimeError, match="Change runtime type"):
        colab.require_gpu("colab", none)
    assert colab.require_gpu("cpu", none) == "cpu"
    t4 = CudaStatus(True, "torch", gpus=[GpuInfo(0, "Tesla T4", 15360, "7.5")])
    assert colab.require_gpu("colab", t4) == "Tesla T4"


def test_run_cli_returns_new_run_dir_and_the_error_line(tmp_path, monkeypatch):
    monkeypatch.setenv(colab.ROOT_ENV, str(tmp_path))
    r = colab.run_cli(["config", "list"], echo=False)
    assert r["returncode"] == 0 and "colab" in r["tail"] and r["run_dir"] is None and r["error"] is None
    r = colab.run_cli(["config", "show", "--profile", "no-such-profile"], echo=False)
    assert r["returncode"] == 2 and r["error"].startswith("error:") and r["run_dir"] is None
    with colab.GpuMonitor() as mon:
        pass
    assert mon.peak_mb is None or mon.peak_mb >= 0


# -- run summaries ---------------------------------------------------------------------------------
META = {"profile": "colab", "device": "cuda", "models": {"asr": "m-asr", "mt": "m-mt"}, "asr_beam": 5, "mt_beam": 4}


def test_speech_rows_flatten_conditions():
    res = {"meta": META, "conditions": {"clean": {
        "n": 2, "errors": 0,
        "asr": {"wer": 0.25, "cer": 0.1, "wer_ci95": {"low": 0.2, "high": 0.3}, "number_words_normalized": {"wer": 0.2},
                "empty_rate": 0.0, "decode_ms": {"median": 1500.0, "p95": 3000.0, "p99": 4000.0}, "rtf": 0.1},
        "speech_to_english": {"bleu": 20.0, "chrf": 48.0, "ci95": {"chrf": [47.0, 49.0]}, "number_recall": {"recall": 0.5}},
        "oracle_mt": {"chrf": 60.0}, "asr_induced_chrf_loss": 12.0}}}
    (row,) = speech_rows(res, "results/eval-speech-1")
    assert row["run"] == "eval-speech-1" and row["device"] == "cuda" and row["wer_ci95"] == "0.2-0.3"
    assert row["decode_p95_s"] == 3.0 and row["s2t_chrf_ci95"] == "47.0-49.0" and row["oracle_chrf"] == 60.0


def test_mt_and_latency_rows_and_missing_values_stay_none(tmp_path):
    mt = {"meta": META, "mt": {"bleu": 39.0, "chrf": 61.6, "ci95": {"bleu": [38, 40], "chrf": [61, 62]},
                              "latency_ms": {"n": 10, "median": 700.0, "p95": 1100.0, "p99": None}}}
    row = mt_row(mt)
    assert row["bleu_ci95"] == "38-40" and row["latency_p50_s"] == 0.7 and row["latency_p99_s"] is None
    assert row["numbers_kept"] is None
    lat = {"meta": META, "latency": {"n_recordings": 3, "latency_from_speech_end_ms": {
        "first_audio": {"median": 3440.0, "p95": 5120.0, "p99": 5700.0}}, "errors": []}}
    row = latency_row(lat)
    assert (row["first_audio_p50_s"], row["first_audio_p99_s"], row["sw_text_p50_s"], row["errors"]) == (3.44, 5.7, None, 0)
    run = tmp_path / "eval-mt-x"
    run.mkdir()
    (run / "results.json").write_text(json.dumps(mt))
    assert summarize_run(run)[0]["chrf"] == 61.6


# -- notebook session ------------------------------------------------------------------------------
def test_interpreter_session_works_inside_a_running_event_loop():
    cfg = load_config("cpu", ["vad.min_silence_ms=300", "vad.speech_pad_ms=0", "vad.hangover_ms=0",
                              "vad.min_speech_ms=60"], environ={})
    backend = FakeBackend()
    backend.load()
    comps = Components(vad=LevelVad(), asr=FakeAsr(), translator=Translator(FakeEngine(), cfg.mt),
                       synthesizer=Synthesizer([backend], cfg.tts))

    async def in_kernel():                     # Jupyter/Colab cells run inside a live loop
        with InterpreterSession(cfg, components=comps) as s:
            r1 = s.interpret(TWO_UTTERANCES, 16000)
            r2 = s.interpret(TWO_UTTERANCES, 16000, realtime=True, block_s=0.05)
        return r1, r2

    r1, r2 = asyncio.run(in_kernel())
    for r in (r1, r2):
        assert len(r["units"]) == 2 and r["english"] and r["swahili"] and not r["errors"]
        assert r["audio"].size > 0 and r["sample_rate"] == cfg.tts.sample_rate
    assert r2["realtime"] and r2["wall_s"] >= r2["input_s"] * 0.9
    assert all(u.get("lat_first_audio_ms") is not None for u in r2["units"])


# -- generated notebooks ---------------------------------------------------------------------------
def _generator():
    spec = importlib.util.spec_from_file_location("make_notebooks", REPO / "scripts" / "make_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_notebooks_are_up_to_date_valid_and_free_of_outputs_and_secrets():
    gen = _generator()
    assert set(gen.NOTEBOOKS) == {p.name for p in (REPO / "notebooks").glob("*.ipynb")}
    for name, cells in gen.NOTEBOOKS.items():
        on_disk = json.loads((REPO / "notebooks" / name).read_text(encoding="utf-8"))
        assert on_disk == gen.nb(name, cells), f"{name} is stale: run python scripts/make_notebooks.py"
        code = [c for c in on_disk["cells"] if c["cell_type"] == "code"]
        assert code and all(c["outputs"] == [] and c["execution_count"] is None for c in code)
        text = json.dumps(on_disk)
        assert "hf_" not in text.replace("hf_login", "").replace("hf_token_set", "")
        assert "colab.bootstrap(" in "".join(code[0]["source"])         # first code cell = shared setup
        for c in code:
            src = "".join(c["source"])
            ast.parse("\n".join("pass" if ln.lstrip().startswith("!") else ln for ln in src.splitlines()))
