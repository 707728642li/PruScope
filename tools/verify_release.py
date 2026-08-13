#!/usr/bin/env python3
"""Fail-closed integrity and privacy checks for the PruScope reviewer repository."""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "SUPPORT.md",
    "AUTHORS.md",
    "MODEL_CARD.md",
    "DATA_CARD.md",
    "RELEASE_READINESS.md",
    "OWNER_ACTION_CHECKLIST.md",
    "docs/INSTALLATION.md",
    "docs/TUTORIAL.md",
    "docs/USER_GUIDE.md",
    "docs/CLI_OR_API_REFERENCE.md",
    "examples/synthetic_metric_audit/predictions.csv",
    "examples/synthetic_metric_audit/expected_summary.json",
    "tests/test_release_integrity.py",
    ".github/workflows/test.yml",
    "release/project_metadata.yml",
    "release/FILE_MANIFEST.csv",
    "release/SHA256SUMS",
)
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".onnx",
    ".engine",
    ".ckpt",
    ".ts",
    ".key",
    ".pem",
    ".p12",
    ".pfx",
}
FORBIDDEN_PARTS = {
    "envs",
    "runs",
    "work",
    "weights",
    "checkpoints",
    "annotations",
    "__pycache__",
}
TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".yml",
    ".yaml",
    ".json",
    ".jsonl",
    ".csv",
    ".cff",
    ".svg",
    ".gitignore",
}
TEXT_FILENAMES = {"LICENSE", "CITATION.cff", ".gitignore"}
TEXT_EXEMPTIONS = {
    "docs/INSTALLATION.md",  # explanatory placeholders such as dataset path error
}
LOGIN_LOCAL_PART = "litaishan" + "910706"
LOGIN_DOMAIN = "gmail" + ".com"
PATTERNS = {
    "login-only email": re.compile(
        re.escape(LOGIN_LOCAL_PART) + r"\s*@\s*" + re.escape(LOGIN_DOMAIN), re.I
    ),
    "Windows absolute path": re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/])"),
    "internal server path": re.compile(r"/(?:data|home)/codexli(?:/|\\)", re.I),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def iter_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    )


def is_text(path: Path) -> bool:
    return path.name in TEXT_FILENAMES or path.suffix.lower() in TEXT_SUFFIXES


def canonical_bytes(path: Path) -> bytes:
    payload = path.read_bytes()
    if is_text(path):
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return payload


def verify_required(errors: list[str]) -> None:
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")


def verify_metadata(errors: list[str]) -> None:
    metadata_text = (ROOT / "release/project_metadata.yml").read_text(encoding="utf-8")
    citation_text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

    def scalar(text: str, key: str) -> str | None:
        match = re.search(rf"(?m)^{re.escape(key)}:\s*(?:\"([^\"]*)\"|'([^']*)'|([^#\r\n]*))", text)
        if not match:
            return None
        value = next((group for group in match.groups() if group is not None), "").strip()
        return value if value else None

    if scalar(metadata_text, "project_name") != "PruScope":
        errors.append("project_metadata project_name must be PruScope")
    if scalar(metadata_text, "public_version") != scalar(citation_text, "version"):
        errors.append("project_metadata and CITATION.cff versions differ")
    expected_url = scalar(metadata_text, "repository")
    if scalar(citation_text, "repository-code") != expected_url:
        errors.append("project_metadata and CITATION.cff repository URLs differ")
    if scalar(metadata_text, "release_tag") not in {None, "null", "~"}:
        errors.append("release_tag must remain null for the private reviewer candidate")
    if scalar(metadata_text, "code_spdx") not in {None, "null", "~"}:
        errors.append("code_spdx must remain null until author/rightsholder approval")


def verify_files(errors: list[str]) -> None:
    for path in iter_files():
        relative = path.relative_to(ROOT).as_posix()
        parts_lower = {part.lower() for part in path.relative_to(ROOT).parts}
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary/credential suffix: {relative}")
        if parts_lower & FORBIDDEN_PARTS:
            errors.append(f"forbidden release directory: {relative}")
        if path.stat().st_size > 50 * 1024 * 1024:
            errors.append(f"file exceeds 50 MiB GitHub review limit: {relative}")
        if is_text(path) and relative not in TEXT_EXEMPTIONS:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"declared text file is not UTF-8: {relative}")
                continue
            for label, pattern in PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{label} detected in {relative}")


def verify_checksums(errors: list[str]) -> None:
    manifest_path = ROOT / "release/FILE_MANIFEST.csv"
    checksum_path = ROOT / "release/SHA256SUMS"
    if not manifest_path.is_file() or not checksum_path.is_file():
        return
    expected: dict[str, tuple[int, str]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            expected[row["path"]] = (int(row["bytes"]), row["sha256"])
    checksum_lines: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        checksum_lines[relative] = digest

    excluded = {"release/FILE_MANIFEST.csv", "release/SHA256SUMS"}
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in iter_files()
        if path.relative_to(ROOT).as_posix() not in excluded
    }
    if set(expected) != actual_paths:
        errors.append("FILE_MANIFEST.csv path set differs from current repository")
    if set(checksum_lines) != actual_paths:
        errors.append("SHA256SUMS path set differs from current repository")
    for relative in sorted(actual_paths & set(expected) & set(checksum_lines)):
        path = ROOT / relative
        payload = canonical_bytes(path)
        digest = hashlib.sha256(payload).hexdigest()
        recorded_size, recorded_digest = expected[relative]
        if len(payload) != recorded_size:
            errors.append(f"manifest size mismatch: {relative}")
        if digest != recorded_digest or digest != checksum_lines[relative]:
            errors.append(f"checksum mismatch: {relative}")


def main() -> int:
    errors: list[str] = []
    verify_required(errors)
    if not errors:
        verify_metadata(errors)
        verify_files(errors)
        verify_checksums(errors)
    if errors:
        print("FAIL: PruScope reviewer repository verification")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS: verified {len(iter_files())} files in PruScope reviewer repository")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
