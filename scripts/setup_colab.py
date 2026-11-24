"""Google Colab setup: install, fetch evaluation data, record the environment, smoke-test.

Run from the repository root in a notebook, after ``src.platform.colab.bootstrap()``:

    !python scripts/setup_colab.py                 # install + data + smoke test
    !python scripts/setup_colab.py --skip-install  # e.g. after a runtime restart
    !python scripts/setup_colab.py --no-data --no-smoke

Idempotent: the install is skipped when this VM already installed the same requirements, data
already on the persistent root (Google Drive) is not downloaded again, and converted models are
reused from the cache. Also works on a local Linux machine (useful to test the script).
Exit code 0 = every smoke check passed (warnings allowed), 1 = a check failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.platform import colab  # noqa: E402


def sh(cmd: list[str], *, check: bool = True, capture: bool = False, timeout: float | None = None):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO, check=check, text=True, timeout=timeout,
                          capture_output=capture, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def requirements_hash() -> str:
    h = hashlib.sha256()
    for f in ("requirements-colab.txt", "pyproject.toml"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:16]


def install(force: bool) -> dict[str, object]:
    marker = Path(tempfile.gettempdir()) / f"swi-setup-{requirements_hash()}.done"
    if marker.exists() and not force:
        return {"skipped": "already installed on this VM", "marker": str(marker)}
    constraints = Path(tempfile.gettempdir()) / "swi-colab-constraints.txt"
    pins = colab.protected_constraints()
    constraints.write_text("\n".join(pins) + "\n", encoding="utf-8")
    t0 = time.perf_counter()
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt", "-c", str(constraints)])
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    return {"seconds": round(time.perf_counter() - t0, 1), "kept": pins}


def fetch_data() -> dict[str, str]:
    out = {}
    if (REPO / "data/flores200/devtest.swh_Latn.txt").exists():
        out["flores200"] = "present"
    else:
        sh([sys.executable, "scripts/fetch_flores200.py"])
        out["flores200"] = "downloaded"
    if (REPO / "data/fleurs_sw/test.jsonl").exists():
        out["fleurs_sw"] = "present"
    else:
        sh([sys.executable, "scripts/fetch_fleurs_sw.py"])
        out["fleurs_sw"] = "downloaded"
    return out


def _cli(args: list[str], profile: str, timeout: float = 1800) -> subprocess.CompletedProcess:
    return sh([sys.executable, "-m", "src.cli", *args, "--profile", profile], check=False, capture=True,
              timeout=timeout)


def smoke(profile: str) -> list[dict[str, str]]:
    """Each stage once, on the configured device. Models are downloaded/converted on first run."""
    checks: list[dict[str, str]] = []

    def add(name: str, ok: bool, detail: str, warn: bool = False) -> None:
        checks.append({"check": name, "status": "ok" if ok else ("warn" if warn else "FAIL"), "detail": detail})

    r = _cli(["config", "show", "--resolve", "--json"], profile, 300)
    if r.returncode == 0:
        res = json.loads(r.stdout)
        rt = res["runtime"]
        add("config + device", True, f"profile {profile}: {rt['device']} {rt.get('gpu_name') or ''} "
            f"{rt['precision']}, ASR {rt['asr_compute_type']}, MT {rt['mt_compute_type']}")
        if rt["device"] != "cuda":
            add("GPU", False, "running on CPU: Runtime > Change runtime type > GPU", warn=True)
        for f in res.get("fallbacks", []):
            add("fallback", False, f"{f}", warn=True)
    else:
        add("config + device", False, r.stderr.strip()[-500:])
        return checks

    r = _cli(["translate", "--text", "Habari za asubuhi. Bei ya mahindi imepanda sokoni leo."], profile)
    add("MT", r.returncode == 0 and bool(r.stdout.strip()), (r.stdout.strip() or r.stderr.strip())[-300:])

    wav = Path(tempfile.gettempdir()) / "swi-smoke-tts.wav"
    r = _cli(["speak", "--text", "The price of maize went up at the market today.", "--output", str(wav)], profile)
    add("TTS", r.returncode == 0 and wav.exists() and wav.stat().st_size > 10_000,
        f"{wav} ({wav.stat().st_size // 1024} KB)" if wav.exists() else r.stderr.strip()[-300:])

    index = REPO / "data/fleurs_sw/test.jsonl"
    if index.exists():
        item = json.loads(index.read_text(encoding="utf-8").splitlines()[0])
        r = _cli(["transcribe", "--input", str(REPO / item["path"]), "--reference", item["raw"], "--quiet"], profile)
        add("ASR (FLEURS clip)", r.returncode == 0, (r.stdout.strip() or r.stderr.strip())[-400:])
        r = _cli(["interpret", "--input", str(REPO / item["path"]), "--quiet", "--no-tts"], profile)
        add("full pipeline", r.returncode == 0, (r.stdout.strip() or r.stderr.strip())[-400:])
    else:
        add("ASR (FLEURS clip)", False, "no FLEURS data (run without --no-data)", warn=True)
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=os.environ.get("SWI_PROFILE", "colab"))
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--force-install", action="store_true")
    ap.add_argument("--no-data", action="store_true")
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()

    if not os.environ.get(colab.ROOT_ENV):
        print(f"note: {colab.ROOT_ENV} is not set; run src.platform.colab.bootstrap() in the notebook first "
              "(otherwise caches go to the default local folder)", file=sys.stderr)
    summary: dict[str, object] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not args.skip_install:
        summary["install"] = install(args.force_install)
    env_dir = Path(os.environ.get(colab.ROOT_ENV, REPO)) / "env"
    summary["lock_file"] = str(colab.write_lock_file(env_dir))
    report = colab.environment_report()
    summary["environment"] = {k: v for k, v in report.items() if k != "nvidia_smi"}
    if report.get("nvidia_smi"):
        print(report["nvidia_smi"])
    if not args.no_data:
        summary["data"] = fetch_data()
    checks = [] if args.no_smoke else smoke(args.profile)
    summary["smoke"] = checks
    path = colab.save_json(summary, env_dir / f"setup-{time.strftime('%Y%m%d-%H%M%S')}.json")

    pk = report["packages"]
    print(f"\npython {report['python']} | torch {pk.get('torch')} | ctranslate2 {pk.get('ctranslate2')} "
          f"| faster-whisper {pk.get('faster-whisper')} | CT2 CUDA devices {report.get('ctranslate2_cuda_devices')}")
    for c in checks:
        print(f"[{c['status']:>4}] {c['check']:<20} {c['detail'].splitlines()[-1] if c['detail'] else ''}")
    print(f"\nenvironment + smoke report: {path}\npackage lock: {summary['lock_file']}")
    return 1 if any(c["status"] == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
