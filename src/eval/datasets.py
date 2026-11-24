"""Evaluation datasets with provenance.

FLEURS Swahili (``scripts/fetch_fleurs_sw.py``) is read speech of FLORES sentences, so each
recording has a human Swahili transcript *and*, through FLORES-200 (``scripts/fetch_flores200.py``),
a human English translation. Recordings are linked to FLORES by exact normalized text (the
FLEURS ids do not follow FLORES line order); unmatched recordings are reported, never guessed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..asr.filters import normalize


class DatasetError(RuntimeError):
    pass


@dataclass
class SpeechItem:
    item_id: str
    audio_path: str
    sw_ref: str
    en_ref: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise DatasetError(f"missing {path} (run scripts/fetch_flores200.py)")
    return path.read_text(encoding="utf-8").splitlines()


def load_flores(root: str | Path = "data/flores200", split: str = "devtest", n: int = 0) -> tuple[list[str], list[str], dict]:
    """(Swahili sources, English human references, provenance) for a FLORES-200 split."""
    root = Path(root)
    src = _read_lines(root / f"{split}.swh_Latn.txt")
    ref = _read_lines(root / f"{split}.eng_Latn.txt")
    if len(src) != len(ref):
        raise DatasetError(f"FLORES {split}: {len(src)} sources vs {len(ref)} references")
    if n:
        src, ref = src[:n], ref[:n]
    man = _manifest(root)
    prov = {"name": "flores200", "split": split, "items": len(src), "subset": f"first {n}" if n else "all",
            "license": man.get("license"), "archive_sha256": man.get("archive_sha256"),
            "references": "human translations"}
    return src, ref, prov


def _manifest(root: Path) -> dict[str, Any]:
    p = root / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def load_fleurs(root: str | Path = "data/fleurs_sw", split: str = "test", n: int = 0,
                flores_root: str | Path | None = "data/flores200") -> tuple[list[SpeechItem], dict]:
    """FLEURS Swahili recordings with human Swahili transcripts and, when FLORES is available,
    human English references. ``n`` takes the first N recordings in manifest order."""
    root = Path(root)
    index = root / f"{split}.jsonl"
    if not index.is_file():
        raise DatasetError(f"missing {index} (run scripts/fetch_fleurs_sw.py)")
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    if n:
        rows = rows[:n]
    english: dict[str, tuple[str, str, int]] = {}
    if flores_root and Path(flores_root).is_dir():
        for fl_split in ("dev", "devtest"):
            try:
                src, ref, _ = load_flores(flores_root, fl_split)
            except DatasetError:
                continue
            for i, (s, e) in enumerate(zip(src, ref, strict=True)):
                english.setdefault(normalize(s), (e, fl_split, i))
    items, unmatched, flores_splits = [], 0, set()
    for r in rows:
        path = Path(r.get("path") or root / split / r["file"])
        hit = english.get(normalize(r["raw"]))
        if hit:
            flores_splits.add(hit[1])
        else:
            unmatched += 1
        items.append(SpeechItem(
            item_id=str(r["file"]).rsplit(".", 1)[0], audio_path=str(path), sw_ref=r["raw"],
            en_ref=hit[0] if hit else None,
            meta={"fleurs_id": r.get("id"), "gender": r.get("gender"),
                  "flores": f"{hit[1]}:{hit[2] + 1}" if hit else None}))
    man = _manifest(root)
    prov = {"name": man.get("name", "fleurs_sw"), "repo": man.get("repo"), "revision": man.get("revision"),
            "archive_sha256": man.get("archive_sha256"), "license": man.get("license"), "split": split,
            "items": len(items), "subset": f"first {n} in manifest order" if n else "all",
            "english_references": f"FLORES-200 {'/'.join(sorted(flores_splits)) or '-'} human translations, "
                                  f"linked by exact normalized text",
            "unmatched_english": unmatched}
    return items, prov


def load_synthetic(root: str | Path = "data/synthetic/sw_tts_v1", n: int = 0) -> tuple[list[SpeechItem], dict]:
    """Synthetic Swahili clips (machine speech with known text; integration evidence only)."""
    root = Path(root)
    man_path = root / "manifest.json"
    if not man_path.is_file():
        raise DatasetError(f"missing {man_path} (run scripts/make_synthetic_swahili.py)")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    rows = man.get("items", [])[: n or None]
    items = [SpeechItem(item_id=r["id"], audio_path=str(root / r["audio"]), sw_ref=r["text"],
                        meta={"voice": r.get("voice"), "sha256": r.get("sha256")}) for r in rows]
    return items, {"name": man.get("name", "sw_tts_v1"), "items": len(items), "generator": man.get("generator"),
                   "kind": "synthetic TTS speech", "english_references": None,
                   "note": "machine speech: easier than real speech; not an accuracy benchmark"}
