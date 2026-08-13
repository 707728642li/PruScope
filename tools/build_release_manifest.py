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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path not in EXCLUDED and ".git" not in path.relative_to(ROOT).parts
    )
    rows = [
        (path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path))
        for path in files
    ]
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
