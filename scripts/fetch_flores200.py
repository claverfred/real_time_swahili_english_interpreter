"""Download FLORES-200 and extract the Swahili/English dev + devtest splits.

    python scripts/fetch_flores200.py

Source: Meta's original release (https://github.com/facebookresearch/flores, CC-BY-SA-4.0).
The archive SHA-256 is recorded in data/flores200/manifest.json. References are independent
professional human translations: never replace them with model output.
Use `dev` for tuning and `devtest` for reporting. NLLB used `dev` during development, so
`devtest` is the fair split for it too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
LANGS = ("swh_Latn", "eng_Latn")
SPLITS = ("dev", "devtest")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/flores200")
    args = ap.parse_args()
    raw, out = Path(args.raw_dir), Path(args.out)
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    archive = raw / "flores200_dataset.tar.gz"
    if not archive.exists():
        print(f"downloading {URL}")
        tmp = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)  # noqa: S310 - fixed https URL
        tmp.replace(archive)
    digest = sha256(archive)

    wanted = {f"flores200_dataset/{s}/{lang}.{s}": (s, lang) for s in SPLITS for lang in LANGS}
    counts = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = member.name.removeprefix("./")
            if name not in wanted or not member.isfile():
                continue  # extract only the named text files (no path traversal possible)
            split, lang = wanted[name]
            lines = tar.extractfile(member).read().decode("utf-8").splitlines()
            (out / f"{split}.{lang}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts[f"{split}.{lang}"] = len(lines)
    missing = [k for k in wanted if f"{wanted[k][0]}.{wanted[k][1]}" not in counts]
    if missing:
        raise SystemExit(f"archive is missing expected files: {missing}")
    for s in SPLITS:
        if counts[f"{s}.swh_Latn"] != counts[f"{s}.eng_Latn"]:
            raise SystemExit(f"{s}: line counts differ")

    manifest = {
        "name": "flores200", "kind": "human_reference_mt", "source_url": URL,
        "archive_sha256": digest, "license": "CC-BY-SA-4.0", "languages": list(LANGS),
        "splits": {s: counts[f"{s}.swh_Latn"] for s in SPLITS},
        "notes": "dev for tuning, devtest for reporting; references are human translations",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
