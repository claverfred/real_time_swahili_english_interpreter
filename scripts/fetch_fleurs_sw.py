"""Download the FLEURS Swahili (sw_ke) test split: real read speech with human transcripts.

    python scripts/fetch_fleurs_sw.py

Source: google/fleurs on the Hugging Face Hub (CC-BY-4.0), pinned revision, archive SHA-256
checked. Writes data/fleurs_sw/test/<id>.wav (16 kHz mono as distributed) and
data/fleurs_sw/test.jsonl with the raw and normalized transcriptions.

Caveat for ASR comparisons: some community Swahili fine-tunes were trained on FLEURS *train*.
The test split is disjoint, but treat such models' FLEURS scores as in-domain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

REPO, REVISION = "google/fleurs", "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
ARCHIVE = "data/sw_ke/audio/test.tar.gz"
ARCHIVE_SHA256 = "8f893fd1f8ab0e53022297a0a89de0454a8a94fcd50c8d2d8784e3dc3261570f"
TSV = "data/sw_ke/test.tsv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/fleurs_sw")
    ap.add_argument("--out", default="data/fleurs_sw")
    args = ap.parse_args()
    raw, out = Path(args.raw_dir), Path(args.out)
    archive = Path(hf_hub_download(REPO, ARCHIVE, repo_type="dataset", revision=REVISION, local_dir=raw))
    tsv = Path(hf_hub_download(REPO, TSV, repo_type="dataset", revision=REVISION, local_dir=raw))
    digest = sha256(archive)
    if digest != ARCHIVE_SHA256:
        raise SystemExit(f"archive checksum mismatch: {digest}")

    # FLEURS tsv columns: id, file name, raw transcription, normalized transcription, chars, n_samples, gender
    rows: dict[str, dict[str, str]] = {}
    with tsv.open(encoding="utf-8", newline="") as fh:
        for r in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows[r[1]] = {"id": r[0], "file": r[1], "raw": r[2], "normalized": r[3], "gender": r[6] if len(r) > 6 else ""}

    wav_dir = out / "test"
    wav_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name          # flatten; never trust archive paths
            if not member.isfile() or not name.endswith(".wav") or name not in rows:
                continue
            (wav_dir / name).write_bytes(tar.extractfile(member).read())
            written.append(name)
    with (out / "test.jsonl").open("w", encoding="utf-8") as fh:
        for name in sorted(written):
            fh.write(json.dumps({**rows[name], "path": str((wav_dir / name).as_posix())}, ensure_ascii=False) + "\n")
    manifest = {"name": "fleurs_sw_ke_test", "kind": "human_transcribed_read_speech", "repo": REPO,
                "revision": REVISION, "archive": ARCHIVE, "archive_sha256": digest, "license": "CC-BY-4.0",
                "utterances": len(written)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
