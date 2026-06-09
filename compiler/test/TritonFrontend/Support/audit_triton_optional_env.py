#!/usr/bin/env python3
"""Audit optional Triton tests/tools for ephemeral interpreter hardcoding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def iter_source_files(repo_root: Path):
    roots = [repo_root / "compiler" / "test", repo_root / "tools"]
    skip_parts = {"build", "__pycache__"}
    suffixes = {".test", ".py", ".sh", ".txt", ".cmake", ""}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or skip_parts.intersection(path.parts):
                continue
            if path.suffix not in suffixes:
                continue
            yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    hardcoded = "/".join([".vc4_auto", "triton_phase6_venv", "bin", "python"])
    violations: list[str] = []
    for path in iter_source_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        if rel == "compiler/test/TritonFrontend/Support/audit_triton_optional_env.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if hardcoded in line:
                violations.append(f"{rel}:{lineno}: {line.strip()}")

    if violations:
        print("TRITON_OPTIONAL_ENV_AUDIT=FAIL")
        print("\n".join(violations))
        return 1
    print("TRITON_OPTIONAL_ENV_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
