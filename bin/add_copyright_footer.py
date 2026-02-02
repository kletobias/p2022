#!/usr/bin/env python3
"""
Add copyright footer to markdown files in _posts/ and _projects/ directories.
Year extracted from filename pattern YYYY-MM-DD- for _posts.
Year extracted from YAML frontmatter date field for _projects.

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
DATE_PATTERN = re.compile(rb"^date:\s[']?(\d{4})-\d{2}-\d{2}[']?$", re.MULTILINE)
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
            raise ValueError(f"No valid date line in frontmatter: {filename}")
        footer = footer_for_year(year)
        if footer in content:
            continue
        targets[filename] = {"year": year, "footer": footer, "dir": PROJECTS_DIR}
    return targets


def validate_directory(
    before: dict[str, bytes],
    after: dict[str, bytes],
    targets: dict[str, dict],
    directory: Path,
    out,
    ts: str,
) -> bool:
    """Validate transformations for a single directory. Returns True on success."""
    dir_name = str(directory)

    # Invariant 1: Same files exist
    if set(before.keys()) != set(after.keys()):
        created = set(after.keys()) - set(before.keys())
        deleted = set(before.keys()) - set(after.keys())
        emit(out, {"ts": ts, "phase": "validate", "dir": dir_name, "check": "same_files", "result": "FAIL", "created": list(created), "deleted": list(deleted)})
        return False
    emit(out, {"ts": ts, "phase": "validate", "dir": dir_name, "check": "same_files", "result": "PASS"})

    # Invariant 2: Non-target files unchanged (byte-for-byte)
    for filename in before:
        if filename in targets:
            continue
        if before[filename] != after[filename]:
            emit(out, {"ts": ts, "phase": "validate", "dir": dir_name, "check": "non_target_unchanged", "file": filename, "result": "FAIL"})
            return False
    emit(out, {"ts": ts, "phase": "validate", "dir": dir_name, "check": "non_target_unchanged", "result": "PASS", "count": len(before) - len(targets)})

    # Invariant 3: Target files = original_stripped + footer (byte-for-byte)
    for filename, info in targets.items():
        original = before[filename]
        original_stripped = original.rstrip()
        footer = info["footer"]
        year = info["year"]
        actual = after[filename]

        # Check 3a: actual starts with original_stripped (byte-for-byte)
        if not actual.startswith(original_stripped):
            emit(out, {"ts": ts, "phase": "validate", "dir": dir_name, "check": "original_preserved", "file": filename, "result": "FAIL"})
            return False

        # Check 3b: actual ends with footer (byte-for-byte)
        if not actual.endswith(footer):
            emit(out, {"ts": ts, "phase": "validate", "dir": dir_name, "check": "footer_appended", "file": filename, "result": "FAIL"})
            return False

        # Check 3c: actual is EXACTLY original_stripped + footer (no extra bytes)
        if actual != original_stripped + footer:
            emit(out, {"ts": ts, "phase": "validate", "dir": dir_name, "check": "exact_concatenation", "file": filename, "result": "FAIL", "actual_len": len(actual), "expected_len": len(original_stripped) + len(footer)})
            return False

        # Check 3d: length difference is exactly footer length
        len_diff = len(actual) - len(original_stripped)
        if len_diff != len(footer):
            emit(out, {"ts": ts, "phase": "validate", "dir": dir_name, "check": "length_diff", "file": filename, "result": "FAIL", "diff": len_diff, "footer_len": len(footer)})
            return False

        # Check 3e: correct year in footer
        year_bytes = f"© Tobias Klein {year}".encode("utf-8")
        if year_bytes not in actual[-len(footer):]:
            emit(out, {"ts": ts, "phase": "validate", "dir": dir_name, "check": "correct_year", "file": filename, "result": "FAIL", "expected_year": year})
            return False

        emit(out, {"ts": ts, "phase": "validate", "dir": dir_name, "file": filename, "result": "PASS", "original_len": len(original), "original_stripped_len": len(original_stripped), "footer_len": len(footer), "actual_len": len(actual)})

    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not POSTS_DIR.is_dir():
        raise ValueError(f"Directory does not exist: {POSTS_DIR}")
    if not PROJECTS_DIR.is_dir():
        raise ValueError(f"Directory does not exist: {PROJECTS_DIR}")

    # Phase 1: Snapshot BEFORE - ground truth for both directories
    before_posts = snapshot_directory(POSTS_DIR)
    before_projects = snapshot_directory(PROJECTS_DIR)

    # Phase 2: Identify targets using dedicated functions
    posts_targets = identify_posts_targets(before_posts)
    projects_targets = identify_projects_targets(before_projects)

    with ARTIFACT_FILE.open("w", encoding="utf-8") as out:
        ts = datetime.now(timezone.utc).isoformat()

        emit(out, {
            "ts": ts,
            "phase": "snapshot_before",
            "posts_file_count": len(before_posts),
            "posts_target_count": len(posts_targets),
            "projects_file_count": len(before_projects),
            "projects_target_count": len(projects_targets),
        })

        # Record ground truth hashes for both directories
        for filename, content in before_posts.items():
            emit(out, {"ts": ts, "phase": "before", "dir": str(POSTS_DIR), "file": filename, "sha256": sha256(content), "len": len(content)})
        for filename, content in before_projects.items():
            emit(out, {"ts": ts, "phase": "before", "dir": str(PROJECTS_DIR), "file": filename, "sha256": sha256(content), "len": len(content)})

        if args.dry_run:
            for filename, info in posts_targets.items():
                original = before_posts[filename]
                footer = info["footer"]
                emit(out, {
                    "ts": ts,
                    "phase": "dry_run",
                    "dir": str(POSTS_DIR),
                    "file": filename,
                    "year": info["year"],
                    "original_len": len(original),
                    "footer_len": len(footer),
                    "expected_len_after": len(original.rstrip()) + len(footer),
                })
            for filename, info in projects_targets.items():
                original = before_projects[filename]
                footer = info["footer"]
                emit(out, {
                    "ts": ts,
                    "phase": "dry_run",
                    "dir": str(PROJECTS_DIR),
                    "file": filename,
                    "year": info["year"],
                    "original_len": len(original),
                    "footer_len": len(footer),
                    "expected_len_after": len(original.rstrip()) + len(footer),
                })
            return 0

        # Phase 3: Apply changes to both directories
        for filename, info in posts_targets.items():
            path = POSTS_DIR / filename
            original = before_posts[filename]
            footer = info["footer"]
            path.write_bytes(original.rstrip() + footer)

        for filename, info in projects_targets.items():
            path = PROJECTS_DIR / filename
            original = before_projects[filename]
            footer = info["footer"]
            path.write_bytes(original.rstrip() + footer)

        # Phase 4: Snapshot AFTER for both directories
        after_posts = snapshot_directory(POSTS_DIR)
        after_projects = snapshot_directory(PROJECTS_DIR)

        # Phase 5: Validate using INVARIANTS
        if not validate_directory(before_posts, after_posts, posts_targets, POSTS_DIR, out, ts):
            return 1
        if not validate_directory(before_projects, after_projects, projects_targets, PROJECTS_DIR, out, ts):
            return 1

        # Record final hashes for both directories
        for filename, content in after_posts.items():
            emit(out, {"ts": ts, "phase": "after", "dir": str(POSTS_DIR), "file": filename, "sha256": sha256(content), "len": len(content)})
        for filename, content in after_projects.items():
            emit(out, {"ts": ts, "phase": "after", "dir": str(PROJECTS_DIR), "file": filename, "sha256": sha256(content), "len": len(content)})

        emit(out, {
            "ts": ts,
            "phase": "complete",
            "posts_modified": len(posts_targets),
            "posts_unchanged": len(before_posts) - len(posts_targets),
            "projects_modified": len(projects_targets),
            "projects_unchanged": len(before_projects) - len(projects_targets),
        })

    return 0


if __name__ == "__main__":
    sys.exit(main())
