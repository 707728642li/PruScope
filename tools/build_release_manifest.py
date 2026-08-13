#!/usr/bin/env python3
"""Regenerate the deterministic file manifest and SHA-256 ledger."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release/FILE_MANIFEST.csv"
CHECKSUMS = ROOT / "release/SHA256SUMS"
EXCLUDED = {MANIFEST, CHECKSUMS}
TEXT_SUFFIXES = {
    ".cff", ".csv", ".gitignore", ".json", ".jsonl", ".md", ".py",
    ".svg", ".txt", ".yaml", ".yml",
}
TEXT_FILENAMES = {"LICENSE", "CITATION.cff", ".gitattributes", ".gitignore"}


def canonical_bytes(path: Path) -> bytes:
    payload = path.read_bytes()
    if path.name in TEXT_FILENAMES or path.suffix.lower() in TEXT_SUFFIXES:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return payload


def main() -> int:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path not in EXCLUDED
        and ".git" not in path.relative_to(ROOT).parts
        and "__pycache__" not in path.relative_to(ROOT).parts
        and path.suffix.lower() not in {".pyc", ".pyo"}
    )
    rows = []
    for path in files:
        payload = canonical_bytes(path)
        rows.append(
            (path.relative_to(ROOT).as_posix(), len(payload), hashlib.sha256(payload).hexdigest())
        )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("path", "bytes", "sha256"))
        writer.writerows(rows)
    CHECKSUMS.write_text(
        "".join(f"{digest}  {relative}\n" for relative, _, digest in rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
