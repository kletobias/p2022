#!/usr/bin/env python3
"""
Add copyright footer to markdown files in _posts/ directory.
Year extracted from filename pattern YYYY-MM-DD-.

Validates using invariants independent of transformation code.
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

POSTS_DIR = Path("_posts")
PROJECTS_DIR = Path("_projects")
ARTIFACT_FILE = Path("/tmp/.copyright_transform.jsonl")
LINKEDIN_URL = "https://www.linkedin.com/in/deep-learning-mastery/"
FILENAME_PATTERN = re.compile(r"^(\d{4})-\d{2}-\d{2}-")
DATE_PATTERN = re.compile(rb"^date:\s*['\"]?(\d{4})", re.MULTILINE)
FOOTER_LEN = 123  # Known constant: len of footer without year + 4 for year


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def footer_for_year(year: str) -> bytes:
    return f"\n\n---\n\n**© Tobias Klein {year} · All rights reserved**<br>\n**LinkedIn: {LINKEDIN_URL}**\n".encode("utf-8")


def snapshot_directory(directory: Path) -> dict[str, bytes]:
    """Capture content of all .md files in directory."""
    return {p.name: p.read_bytes() for p in directory.glob("*.md")}


def extract_year_from_frontmatter(content: bytes) -> str | None:
    """Extract year from YAML frontmatter date field.

    Frontmatter requirements:
    - Line 0 must be exactly '---'
    - Closes at first subsequent line that is exactly '---'
    - date: field searched only within extracted frontmatter block
    """
    lines = content.split(b"\n")

    # Line 0 must be exactly ---
    if not lines or lines[0].rstrip() != b"---":
        return None

    # Find closing --- (first occurrence after line 0)
    closing_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.rstrip() == b"---":
            closing_idx = i
            break

    if closing_idx is None:
        return None

    # Extract frontmatter content (lines 1 to closing_idx-1)
    frontmatter_lines = lines[1:closing_idx]
    frontmatter = b"\n".join(frontmatter_lines)

    # Search for date: at start of line within frontmatter only
    date_match = DATE_PATTERN.search(frontmatter)
    if not date_match:
        return None
    return date_match.group(1).decode("utf-8")


def emit(out, rec: dict) -> None:
    out.write(json.dumps(rec) + "\n")


def identify_posts_targets(snapshot: dict[str, bytes]) -> dict[str, dict]:
    """Identify targets in _posts using filename pattern YYYY-MM-DD-."""
    targets = {}
    for filename, content in snapshot.items():
        if not filename.endswith(".md"):
            continue
        match = FILENAME_PATTERN.match(filename)
        if not match:
            continue
        year = match.group(1)
        footer = footer_for_year(year)
        if footer in content:
            continue
        targets[filename] = {"year": year, "footer": footer, "dir": POSTS_DIR}
    return targets


def identify_projects_targets(snapshot: dict[str, bytes]) -> dict[str, dict]:
    """Identify targets in _projects using frontmatter date field."""
    targets = {}
    for filename, content in snapshot.items():
        if not filename.endswith(".md"):
            continue
        year = extract_year_from_frontmatter(content)
        if year is None:
            continue
        footer = footer_for_year(year)
        if footer in content:
            continue
        targets[filename] = {"year": year, "footer": footer, "dir": PROJECTS_DIR}
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not POSTS_DIR.is_dir():
        return 1

    # Phase 1: Snapshot BEFORE - this is ground truth
    before = snapshot_directory(POSTS_DIR)

    # Phase 2: Identify files to modify (without computing expected)
    targets = {}
    for filename, content in before.items():
        if not filename.endswith(".md"):
            continue
        match = FILENAME_PATTERN.match(filename)
        if not match:
            continue
        year = match.group(1)
        footer = footer_for_year(year)
        if footer in content:
            continue
        targets[filename] = {"year": year, "footer": footer}

    with ARTIFACT_FILE.open("w", encoding="utf-8") as out:
        ts = datetime.now(timezone.utc).isoformat()

        emit(out, {"ts": ts, "phase": "snapshot_before", "file_count": len(before), "target_count": len(targets)})

        # Record ground truth hashes
        for filename, content in before.items():
            emit(out, {"ts": ts, "phase": "before", "file": filename, "sha256": sha256(content), "len": len(content)})

        if args.dry_run:
            for filename, info in targets.items():
                original = before[filename]
                footer = info["footer"]
                emit(out, {
                    "ts": ts,
                    "phase": "dry_run",
                    "file": filename,
                    "year": info["year"],
                    "original_len": len(original),
                    "footer_len": len(footer),
                    "expected_len_after": len(original.rstrip()) + len(footer),
                })
            return 0

        # Phase 3: Apply changes
        for filename, info in targets.items():
            path = POSTS_DIR / filename
            original = before[filename]
            footer = info["footer"]
            path.write_bytes(original.rstrip() + footer)

        # Phase 4: Snapshot AFTER
        after = snapshot_directory(POSTS_DIR)

        # Phase 5: Validate using INVARIANTS (not computed expected)

        # Invariant 1: Same files exist
        if set(before.keys()) != set(after.keys()):
            created = set(after.keys()) - set(before.keys())
            deleted = set(before.keys()) - set(after.keys())
            emit(out, {"ts": ts, "phase": "validate", "check": "same_files", "result": "FAIL", "created": list(created), "deleted": list(deleted)})
            return 1
        emit(out, {"ts": ts, "phase": "validate", "check": "same_files", "result": "PASS"})

        # Invariant 2: Non-target files unchanged (byte-for-byte)
        for filename in before:
            if filename in targets:
                continue
            if before[filename] != after[filename]:
                emit(out, {"ts": ts, "phase": "validate", "check": "non_target_unchanged", "file": filename, "result": "FAIL"})
                return 1
        emit(out, {"ts": ts, "phase": "validate", "check": "non_target_unchanged", "result": "PASS", "count": len(before) - len(targets)})

        # Invariant 3: Target files = original_stripped + footer (byte-for-byte)
        for filename, info in targets.items():
            original = before[filename]
            original_stripped = original.rstrip()
            footer = info["footer"]
            year = info["year"]
            actual = after[filename]

            # Check 3a: actual starts with original_stripped (byte-for-byte)
            if not actual.startswith(original_stripped):
                emit(out, {"ts": ts, "phase": "validate", "check": "original_preserved", "file": filename, "result": "FAIL"})
                return 1

            # Check 3b: actual ends with footer (byte-for-byte)
            if not actual.endswith(footer):
                emit(out, {"ts": ts, "phase": "validate", "check": "footer_appended", "file": filename, "result": "FAIL"})
                return 1

            # Check 3c: actual is EXACTLY original_stripped + footer (no extra bytes)
            if actual != original_stripped + footer:
                emit(out, {"ts": ts, "phase": "validate", "check": "exact_concatenation", "file": filename, "result": "FAIL", "actual_len": len(actual), "expected_len": len(original_stripped) + len(footer)})
                return 1

            # Check 3d: length difference is exactly footer length
            len_diff = len(actual) - len(original_stripped)
            if len_diff != len(footer):
                emit(out, {"ts": ts, "phase": "validate", "check": "length_diff", "file": filename, "result": "FAIL", "diff": len_diff, "footer_len": len(footer)})
                return 1

            # Check 3e: correct year in footer
            year_bytes = f"© Tobias Klein {year}".encode("utf-8")
            if year_bytes not in actual[-len(footer):]:
                emit(out, {"ts": ts, "phase": "validate", "check": "correct_year", "file": filename, "result": "FAIL", "expected_year": year})
                return 1

            emit(out, {"ts": ts, "phase": "validate", "file": filename, "result": "PASS", "original_len": len(original), "original_stripped_len": len(original_stripped), "footer_len": len(footer), "actual_len": len(actual)})

        # Record final hashes
        for filename, content in after.items():
            emit(out, {"ts": ts, "phase": "after", "file": filename, "sha256": sha256(content), "len": len(content)})

        emit(out, {"ts": ts, "phase": "complete", "modified": len(targets), "unchanged": len(before) - len(targets)})

    return 0


if __name__ == "__main__":
    sys.exit(main())
