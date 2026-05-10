#!/usr/bin/env python3
"""Build bounded GPT Pro context packs for VC4 codegen Milestone 1.

The context packer is deliberately deterministic: profiles in
pro_scripts/vc4_codegen_m1_context_profiles.json select which project files,
fixtures, summaries, and failure artifacts are included.  Missing future files
are recorded as notes instead of causing prompt generation to fail, so the same
profiles can be used before and after later slices create new compiler code.

This script never modifies the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from vc4_codegen_state import (
        DriverError,
        MilestoneConfig,
        find_repo_root,
        git_changed_paths,
        read_json_file,
        relpath,
        tail_file,
        write_json_file,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vc4_codegen_state import (  # type: ignore
        DriverError,
        MilestoneConfig,
        find_repo_root,
        git_changed_paths,
        read_json_file,
        relpath,
        tail_file,
        write_json_file,
    )


DEFAULT_PER_FILE_CHAR_LIMIT = 26000
DEFAULT_FOCUSED_FILE_CHAR_LIMIT = 42000
DEFAULT_TEST_FILE_CHAR_LIMIT = 30000
DEFAULT_EXTRACTOR_CHAR_LIMIT = 50000

# Text artifact suffixes that are safe/useful to include verbatim in GPT
# context.  The no-truncation patch made several extractors include complete
# generated artifacts and changed files; this constant must be module-level so
# those extractors do not fail at prompt-render time.
TEXT_EXTS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
    ".td", ".mlir", ".qasm", ".json", ".py", ".sh", ".md", ".txt",
    ".log", ".cfg", ".in", ".cmake", ".mk", ".make", "",
}


@dataclass
class Section:
    title: str
    body: str
    source: str = ""

    def render(self) -> str:
        source = f"\n_Source: {self.source}_" if self.source else ""
        body = self.body.rstrip() or "<empty>"
        return f"\n\n## {self.title}{source}\n\n{body}\n"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_optional_int(value: Any, default: int = 0) -> int:
    """Parse an optional integer-ish config value.

    Context max_chars is ignored by this workflow, but profiles may still carry
    legacy values such as "full".  Metadata generation should not crash on
    those legacy strings.
    """
    if value is None or value == "":
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "default"}:
            return default
        if lowered in {"full", "none", "unlimited", "all"}:
            return 0
    try:
        return int(value)
    except Exception:
        return default


def clean_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def fenced(text: str, *, language: str = "") -> str:
    # Use a long fence to avoid accidental closure if source contains ```.
    fence = "````"
    lang = language.strip()
    return f"{fence}{lang}\n{text.rstrip()}\n{fence}"



def soft_truncate(text: str, limit: int, *, marker: str = "[... truncated ...]") -> str:
    """Return complete text.

    The VC4 workflow intentionally does not truncate prompt context anymore.
    Earlier budget trimming caused GPT to see partial source files and then try
    to reconstruct missing code.  The `limit` and `marker` parameters remain for
    backward-compatible callers but are deliberately ignored.
    """
    return clean_text(text)

def read_text(path: Path, *, limit: int | None = None) -> str:
    """Read a complete text file; `limit` is retained as a no-op API arg."""
    return clean_text(path.read_text(encoding="utf-8", errors="replace"))
def item_char_limit(item: Mapping[str, Any], default_limit: int | None) -> int | None:
    """Return per-item char limit. Use max_chars="full" for full text."""
    raw = item.get("max_chars", item.get("limit_chars", None))
    if raw is None:
        return default_limit
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"", "default"}:
            return default_limit
        if value in {"full", "none", "unlimited", "all"}:
            return None
        return int(value)
    value = int(raw)
    return None if value <= 0 else value


def language_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".cpp", ".cc", ".cxx", ".h", ".hpp"}:
        return "cpp"
    if suffix in {".td", ".mlir", ".qasm"}:
        return suffix[1:]
    if suffix == ".json":
        return "json"
    if suffix == ".py":
        return "python"
    if suffix == ".sh":
        return "bash"
    if suffix == ".md":
        return "markdown"
    return "text"


def run_git(repo: Path, args: Sequence[str]) -> tuple[int, str, str]:
    proc = subprocess.run(["git", *args], cwd=str(repo), text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def git_diff_stat(repo: Path) -> str:
    rc, out, err = run_git(repo, ["diff", "--stat"])
    unstaged = out.rstrip()
    rc2, out2, err2 = run_git(repo, ["diff", "--cached", "--stat"])
    staged = out2.rstrip()
    parts: list[str] = []
    if staged:
        parts.append("### staged diff --stat\n" + staged)
    if unstaged:
        parts.append("### unstaged diff --stat\n" + unstaged)
    if not parts:
        return "No staged or unstaged git diff."
    return "\n\n".join(parts)


def git_diff_limited(repo: Path, paths: Sequence[str] | None = None, *, limit: int = 50000) -> str:
    cmd = ["diff", "--"] + list(paths or [])
    rc, out, err = run_git(repo, cmd)
    if rc != 0:
        return f"git diff failed:\n{err}"
    if not out.strip():
        return "No unstaged git diff for selected paths."
    return soft_truncate(out, limit)



def repo_tree_excerpt(repo: Path, roots: Sequence[str], *, max_files: int = 0) -> str:
    """Return a complete deterministic tree listing for the requested roots.

    `max_files` is ignored unless explicitly positive and the environment sets
    VC4_CONTEXT_ENABLE_TREE_LIMIT=1.  Normal GPT prompts should not silently omit
    files from the selected context.
    """
    rows: list[str] = []
    skipped = 0
    limit_enabled = os.environ.get("VC4_CONTEXT_ENABLE_TREE_LIMIT", "").lower() in {"1", "true", "yes"}
    for root in roots:
        root_path = repo / root
        if not root_path.exists():
            rows.append(f"<missing> {root}")
            continue
        if root_path.is_file():
            rows.append(root)
            continue
        for path in sorted(root_path.rglob("*")):
            rel = relpath(repo, path)
            if any(part in {".git", ".vc4_auto", "build", "__pycache__"} for part in path.parts):
                continue
            if path.is_dir():
                continue
            if limit_enabled and max_files > 0 and len(rows) >= max_files:
                skipped += 1
                continue
            rows.append(rel)
    if skipped:
        rows.append(f"... {skipped} more file(s) omitted because VC4_CONTEXT_ENABLE_TREE_LIMIT=1 ...")
    return "\n".join(rows) if rows else "<empty tree excerpt>"
def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged



def focused_excerpt(text: str, focus_terms: Sequence[str], *, window: int = 70, limit: int = DEFAULT_FOCUSED_FILE_CHAR_LIMIT) -> str:
    """Return the complete file, not a focus-window excerpt.

    Focus terms are still reported by include_file_excerpt(), but source code is
    sent whole.  This avoids the failure mode where GPT sees a truncated
    VC4ArtifactEmitter.cpp and invents replacement functions.
    """
    return clean_text(text)
def include_file(repo: Path, item: Mapping[str, Any], *, default_limit: int | None = DEFAULT_PER_FILE_CHAR_LIMIT) -> Section:
    raw_path = str(item.get("path", ""))
    path = repo / raw_path
    if not path.exists():
        return Section("Missing file", f"Requested file does not exist yet: `{raw_path}`", raw_path)
    # The no-truncation workflow intentionally ignores per-file limits.  Keep
    # the API call for compatibility, but report the truth in the prompt.
    item_char_limit(item, default_limit)
    text = read_text(path)
    return Section(
        f"File: {raw_path}",
        "FULL FILE INCLUDED. No context/profile max_chars budget was applied.\n\n"
        + fenced(text, language=language_for(path)),
        raw_path,
    )



def include_file_excerpt(repo: Path, item: Mapping[str, Any]) -> Section:
    raw_path = str(item.get("path", ""))
    path = repo / raw_path
    focus = [str(x) for x in item.get("focus", [])]
    if not path.exists():
        return Section("Missing file excerpt", f"Requested file does not exist yet: `{raw_path}`", raw_path)
    text = read_text(path)
    focus_line = "Focus terms retained for navigation only; FULL FILE INCLUDED: " + ", ".join(focus) if focus else "FULL FILE INCLUDED."
    return Section(f"Full file formerly excerpted: {raw_path}", focus_line + "\n\n" + fenced(text, language=language_for(path)), raw_path)
def include_tests(repo: Path, item: Mapping[str, Any]) -> list[Section]:
    sections: list[Section] = []
    for raw in item.get("paths", []):
        raw_path = str(raw)
        path = repo / raw_path
        if not path.exists():
            sections.append(Section("Missing test", f"Requested test does not exist yet: `{raw_path}`", raw_path))
            continue
        text = read_text(path, limit=DEFAULT_TEST_FILE_CHAR_LIMIT)
        sections.append(Section(f"Test: {raw_path}", fenced(text, language=language_for(path)), raw_path))
    return sections


def include_fixture_tree(repo: Path, item: Mapping[str, Any]) -> Section:
    raw_path = str(item.get("path", ""))
    root = repo / raw_path
    if not root.exists():
        return Section("Missing fixture tree", f"Requested fixture tree does not exist yet: `{raw_path}`", raw_path)
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        lines.append(f"{relpath(repo, path)}\t{size} bytes")
    return Section(f"Fixture tree: {raw_path}", "\n".join(lines) if lines else "<empty>", raw_path)



def include_generated_artifacts(repo: Path, item: Mapping[str, Any], slice_entry: Mapping[str, Any], failure_packet: Mapping[str, Any] | None) -> list[Section]:
    """Include complete generated artifacts from candidate bundles.

    This scans both M1 and M2 candidate roots plus any bundle/output/staging
    paths named in the failure packet.  Text artifacts are included whole.
    """
    sections: list[Section] = []
    configured = [str(p) for p in item.get("paths", [])]
    roots: list[Path] = []
    seen_roots: set[str] = set()

    def add_root(path: Path) -> None:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen_roots:
            seen_roots.add(key)
            roots.append(path)

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in {"bundle", "candidate_dir", "output_dir", "staging_dir"} and isinstance(value, str) and value:
                    add_root(value and ((repo / value) if not Path(value).is_absolute() else Path(value)))
                visit(value)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    fixture_names = {
        "minimal_thrend", "memory_output", "read_nop_write", "saxpy_full", "multi_kernel_chain",
        "saxpy_16", "saxpy_basic", "global_store_coalesced_multi", "gemv_naive_tail",
        "saxpy_tmu", "saxpy_tmu_overlap", "tmu_read_nop_write", "tmu_strided_load", "sfu_recip",
        "vpm_setup_clobber", "vpm_slice_visibility", "qpu_barrier_syncthreads", "warp_reduce_sum",
        "warp_prefix_sum", "block_reduce_sum", "shared_transpose_16x16",
    }
    if failure_packet:
        visit(failure_packet)
        for key in ("fixture", "name"):
            def gather(node: Any) -> None:
                if isinstance(node, Mapping):
                    for k, v in node.items():
                        if k == key and isinstance(v, str) and v:
                            fixture_names.add(v)
                        gather(v)
                elif isinstance(node, list):
                    for child in node:
                        gather(child)
            gather(failure_packet)

    for state_root in [repo / ".vc4_auto/codegen_m1/candidates", repo / ".vc4_auto/codegen_m2/candidates"]:
        for name in sorted(fixture_names):
            add_root(state_root / name)

    patterns = configured or [
        "manifest.json", "layout.json", "kernel_launch.c", "kernel_launch.h",
        "kernels/*.qasm", "assembled/*.c", "assembled/*.h", "*.qasm", "*_shader.c", "*_shader.h",
        "*.json", "*.c", "*.h",
    ]
    seen_paths: set[str] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in TEXT_EXTS:
                    continue
                key = str(path.resolve())
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                sections.append(Section(f"Generated artifact: {relpath(repo, path)}", fenced(read_text(path), language=language_for(path)), relpath(repo, path)))
    if not sections:
        sections.append(Section("Generated artifacts", "No generated candidate artifacts were found yet under M1/M2 candidate roots or failure packet paths."))
    return sections
def format_slice(slice_entry: Mapping[str, Any], config: MilestoneConfig) -> str:
    fields = {
        "id": slice_entry.get("id"),
        "title": slice_entry.get("title"),
        "intent": slice_entry.get("intent"),
        "depends_on": slice_entry.get("depends_on", []),
        "context_profile": slice_entry.get("context_profile"),
        "gates": slice_entry.get("gates", []),
        "allowed_paths": config.allowed_paths_for_slice(slice_entry),
        "forbidden_paths": config.forbidden_paths_for_slice(slice_entry),
        "codex_policy": slice_entry.get("codex_policy"),
        "max_gpt_attempts": slice_entry.get("max_gpt_attempts"),
        "max_codex_attempts": slice_entry.get("max_codex_attempts"),
        "commit_message": slice_entry.get("commit_message"),
        "non_goals": slice_entry.get("non_goals", []),
    }
    return fenced(json.dumps(fields, indent=2, sort_keys=True), language="json")


def static_summary(name: str) -> str:
    summaries: dict[str, str] = {
        "launch_abi_summary": """
Milestone 1 launch ABI facts to preserve:
- `vc4.launch_abi` is a dictionary attribute on a kernel `vc4.func` in QPU domain.
- It requires non-empty `public_name`, `tail_policy` of `exact_multiple` or `tail_safe`, positive `uniform_words_per_qpu`, `args`, and `builtins`.
- Scalar args are semantic public arguments with direction `by_value` and type one of `i32`, `u32`, `f32`, or `index`.
- Buffer args are semantic public arguments with direction `in`, `out`, or `inout` and element type among `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `f32`.
- Builtins may be uniform suffixes; `num_qpus` must use `materialization = "uniform_suffix"`; `elem_num` must not appear in launch ABI.
- All uniform indices from args and uniform-suffix builtins must be unique and dense in `[0, uniform_words_per_qpu)`.
- Generated public launcher APIs must not expose raw uniform arrays, `qpu_id`, or `num_qpus`; those are internal launcher/runtime details.
""".strip(),
        "scheduled_sink_summary": """
Milestone 1 qasm input is final-stage scheduled QPU sink IR only:
- exactly one launchable QPU kernel function is targeted per `vc4-codegen` invocation;
- function domain must be `#vc4.execution_domain<qpu>`;
- function form must be `#vc4.function_form<scheduled>`;
- function must have `kernel` and `vc4.launch_abi`;
- body may contain only `vc4.qpu.bundle`, `vc4.qpu.ldi`, `vc4.qpu.sema`, and `vc4.qpu.branch`;
- structured VC4 ops and GPU lowering are non-goals for Milestone 1.
""".strip(),
        "verify_emit_contract_summary": """
The existing emit-contract verifier is the front-end guard for artifact emission. It rejects structured QPU functions as not directly emittable, rejects host scheduled functions as launcher-only mismatches, and permits only scheduled qasm sink ops in scheduled QPU functions. It also requires the flattened scheduled instruction stream to end with exactly an explicit `thrend` instruction in slot N-3 followed by two non-branch scheduled delay-slot instructions in slots N-2 and N-1.
""".strip(),
        "scheduled_epilogue_summary": """
For qasm emission, the final flattened scheduled stream must end:
1. `vc4.qpu.bundle` with `sig = #vc4.qpu_signal<thrend>`
2. non-branch scheduled op delay slot
3. non-branch scheduled op delay slot
Generated qasm for the minimal smoke should canonicalize this as `thrend`, `nop`, `nop` if the input ops encode the no-op delay slots.
""".strip(),
        "qpu_bundle_verifier_summary": """
`vc4.qpu.bundle` represents an ALU/signal slot. Important verifier constraints: exactly one of `raddr_b` or `small_imm`; `small_imm` requires `sig = small_imm`; dedicated load-immediate/branch signals are represented by `vc4.qpu.ldi`/`vc4.qpu.branch`; write addresses are in [0,63]; read addresses are in [0,63]; pack/unpack attribute families depend on `pm`; active ADD and MUL pipes may not target the same accumulator/I/O write address.
""".strip(),
        "qpu_ldi_verifier_summary": """
`vc4.qpu.ldi` is the scheduled load-immediate instruction. `splat32` requires a signless i32 `value`; per-element modes require a dense i32 array of exactly 16 lane values in the mode-specific range. Pack family depends on `pm`; write addresses are in [0,63].
""".strip(),
        "qpu_sema_verifier_summary": """
`vc4.qpu.sema` is the scheduled semaphore instruction. It requires id in [0,15], pack family according to `pm`, write addresses in [0,63], and rejects stall-capable peripheral write addresses for semaphore slots.
""".strip(),
        "qpu_branch_verifier_summary": """
`vc4.qpu.branch` has an explicit delay-slot region with exactly one block, no block arguments, and exactly three scheduled QPU ops. Branch read address `raddr_a` is in [0,31]; write addresses are in [0,63]. Emission should preserve explicit delay-slot order and generate deterministic labels.
""".strip(),
        "scheduled_instruction_stream_flattening": """
The scheduled instruction stream is flattened in top-level program order. A `vc4.qpu.branch` contributes the branch instruction itself followed immediately by the three operations in its explicit delay-slot region. Cross-instruction verifiers reason over this flattened stream, so qasm emission should use the same flattening.
""".strip(),
        "hardware_fixture_summary": """
Hardware fixture inputs are source truth: input.mlir, run.sh, candidate/run.sh, reference bundles, and expected.json should be used to understand the fixture. Do not mutate reference bundles or expected.json unless a slice explicitly allows it. Candidate generated artifacts and verifier logs are the debugging targets.
""".strip(),
        "hardware_contract_summary": """
Hardware-run tests have a reference side and a future candidate side. The reference side is immutable ground truth. Candidate codegen is expected to generate `kernel.qasm`, `kernel_launch.c`, and `kernel_launch.h` from `input.mlir`. Passing is determined by the semantic oracle and `expected.json`, not by exact qasm text matching.
""".strip(),
        "candidate_vs_reference_summary": """
Reference bundles under `reference/` must not be edited by the codegen workflow. Candidate bundles may be regenerated under a candidate/staging directory. The candidate runner should assemble, build, run, and compare candidate output against the same expected-result oracle used by the reference.
""".strip(),
        "expected_json_schema": """
The expected-result schema observed so far uses top-level `name` and `status`; exact required fields under `required`; and optional tolerance maps such as `float_max`. The checker should require `status = PASS` unless the test explicitly says otherwise, compare `required` fields exactly, and apply tolerance groups only where present.
""".strip(),
        "generated_artifact_summary": """
When diagnosing candidate failures, include generated `kernel.qasm`, `kernel_launch.c`, `kernel_launch.h`, `manifest.json`, build logs, assembly logs, run logs, and any parsed `VC4_TEST_RESULT`. The goal is root-cause diagnosis of generated artifacts, not mutation of reference bundles.
""".strip(),
        "vc4_dialect_registration": """
`vc4-opt` registers the VC4 dialect and verifier passes with MLIR's optimizer driver. The Milestone 1 artifact tool should similarly register the VC4 dialect, parse an MLIR file, optionally run the required verifier pipeline, and then call reusable Target/VC4 emission libraries.
""".strip(),
    }
    return summaries.get(name, f"No static summary is registered for extractor `{name}`.")


def extractor_cmake_neighbors(repo: Path) -> str:
    candidates = [
        "compiler/tools/CMakeLists.txt",
        "compiler/tools/vc4-opt/CMakeLists.txt",
        "compiler/lib/CMakeLists.txt",
        "compiler/lib/Dialect/VC4/CMakeLists.txt",
        "compiler/lib/Dialect/VC4/IR/CMakeLists.txt",
        "compiler/include/vc4/CMakeLists.txt",
        "compiler/include/vc4/Dialect/VC4/IR/CMakeLists.txt",
        "compiler/test/CodeGen/VC4/CMakeLists.txt",
    ]
    parts: list[str] = []
    for raw in candidates:
        path = repo / raw
        if path.exists():
            parts.append(f"### {raw}\n" + fenced(read_text(path, limit=12000), language=language_for(path)))
        else:
            parts.append(f"### {raw}\n<missing>")
    return "\n\n".join(parts)


def extractor_minimal_input(repo: Path) -> str:
    path = repo / "compiler/test/CodeGen/VC4/Hardware/Run/minimal_thrend/input.mlir"
    if not path.exists():
        return "minimal_thrend input.mlir is not present."
    return fenced(read_text(path, limit=24000), language="mlir")


def extractor_minimal_qasm(repo: Path) -> str:
    path = repo / "compiler/test/CodeGen/VC4/Hardware/Run/minimal_thrend/reference/minimal_thrend.qasm"
    if not path.exists():
        return "minimal_thrend reference qasm is not present."
    return fenced(read_text(path, limit=12000), language="qasm")



def extractor_existing_codegen_files(repo: Path) -> str:
    roots = [
        "compiler/include/vc4/Target/VC4",
        "compiler/lib/Target/VC4",
        "compiler/tools/vc4-codegen",
        "compiler/test/CodeGen/VC4/Emit",
        "compiler/test/CodeGen/VC4/Support",
    ]
    files: list[Path] = []
    missing: list[str] = []
    for raw in roots:
        root = repo / raw
        if not root.exists():
            missing.append(raw)
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in TEXT_EXTS:
                files.append(path)
    parts: list[str] = []
    if missing:
        parts.append("Missing/not-yet-created codegen roots:\n" + "\n".join(f"- {m}" for m in missing))
    if files:
        parts.append("Existing codegen-related files:\n" + "\n".join(f"- {relpath(repo, f)}" for f in files))
        for path in files:
            parts.append(f"### {relpath(repo, path)}\n" + fenced(read_text(path), language=language_for(path)))
    else:
        parts.append("No existing Target/VC4 codegen files were found yet.")
    return "\n\n".join(parts)
def extractor_git_diff_summary(repo: Path) -> str:
    changed = git_changed_paths(repo, include_untracked=True)
    parts = [git_diff_stat(repo)]
    if changed:
        parts.append("Changed/untracked paths:\n" + "\n".join(f"- {p}" for p in changed[:200]))
    return "\n\n".join(parts)



def extractor_changed_files_excerpt(repo: Path, slice_entry: Mapping[str, Any], config: MilestoneConfig) -> str:
    changed = git_changed_paths(repo, include_untracked=True)
    allowed = config.allowed_paths_for_slice(slice_entry)
    parts: list[str] = []
    for raw in changed:
        path = repo / raw
        if not path.exists() or path.is_dir():
            continue
        if not any(fnmatch_path(raw, pat) for pat in allowed):
            continue
        if path.suffix.lower() not in TEXT_EXTS:
            continue
        parts.append(f"### {raw}\n" + fenced(read_text(path), language=language_for(path)))
    return "\n\n".join(parts) if parts else "No changed allowed text files to include."
def fnmatch_path(path: str, pattern: str) -> bool:
    import fnmatch
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, pattern.rstrip("/") + "/**")


def sanitize_failure_packet_for_prompt(data: Any) -> Any:
    if isinstance(data, list):
        return [sanitize_failure_packet_for_prompt(x) for x in data]
    if not isinstance(data, dict):
        return data
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key == "apply_bundle_sh":
            out[key] = "<redacted: previous generated apply shell; do not copy; use current DOWNLOAD_CONTRACT_JSON and trusted applier>"
            continue
        if key == "artifact_transport_json":
            summary: dict[str, Any] = {"redacted": True, "reason": "previous attempt artifact metadata contains stale filenames"}
            if isinstance(value, dict):
                for keep in ("schema_version", "transport", "collected_at"):
                    if keep in value:
                        summary[keep] = value[keep]
                for name_key in ("artifact_prefix", "bundle_zip", "apply_script"):
                    if name_key in value:
                        summary[f"{name_key}_redacted"] = True
            out[key] = summary
            continue
        if key == "bundle_manifest_json":
            summary: dict[str, Any] = {"redacted": True, "reason": "previous bundle manifest may contain stale attempt IDs and filenames"}
            if isinstance(value, dict):
                for keep in ("schema_version", "transport", "slice_id", "attempt"):
                    if keep in value:
                        summary[keep] = value[keep]
                changed = value.get("changed_paths")
                if isinstance(changed, list):
                    summary["changed_paths_count"] = len(changed)
                    summary["changed_paths_preview"] = [(x.get("path") if isinstance(x, dict) else x) for x in changed[:20]]
                if "diagnosis" in value:
                    summary["diagnosis_type"] = type(value.get("diagnosis")).__name__
                    if not isinstance(value.get("diagnosis"), list):
                        summary["diagnosis_error"] = "diagnosis must be an array"
            out[key] = summary
            continue
        if key == "bundle_zip":
            out[key] = {"path_redacted": True, "bytes": value.get("bytes") if isinstance(value, dict) else None}
            continue
        if key == "bundle_zip_members":
            out[key] = "<redacted: previous bundle member list>"
            continue
        out[key] = sanitize_failure_packet_for_prompt(value)
    if data.get("stage") == "patch-guard" or "extra" in data:
        out["artifact_filename_rule"] = "Do not copy artifact filenames from this failure packet; use only the current prompt's DOWNLOAD_CONTRACT_JSON."
    return out



def extractor_failure_packet(repo: Path, failure_packet_path: Path | None) -> tuple[str, Mapping[str, Any] | None]:
    if not failure_packet_path:
        return "No failure packet was provided.", None
    path = failure_packet_path if failure_packet_path.is_absolute() else repo / failure_packet_path
    if not path.exists():
        return f"Failure packet path does not exist: `{path}`", None
    try:
        data = read_json_file(path)
    except Exception as exc:
        return f"Could not parse failure packet `{relpath(repo, path)}`: {exc}", None
    sanitized = sanitize_failure_packet_for_prompt(data)
    body = fenced(json.dumps(sanitized, indent=2, sort_keys=True), language="json")

    log_paths: list[Path] = []
    seen: set[str] = set()
    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                if k == "log_path" and isinstance(v, str) and v:
                    p = Path(v)
                    if not p.is_absolute():
                        p = repo / p
                    key = str(p.resolve()) if p.exists() else str(p)
                    if key not in seen:
                        seen.add(key)
                        log_paths.append(p)
                visit(v)
        elif isinstance(node, list):
            for child in node:
                visit(child)
    visit(data)
    if log_paths:
        body += "\n\n### Full referenced logs"
        for lp in log_paths:
            if lp.exists():
                body += f"\n\n#### {relpath(repo, lp)}\n" + fenced(read_text(lp), language="text")
            else:
                body += f"\n\n#### {lp}\n<missing>"
    return body, data if isinstance(data, dict) else None
def run_extractor(name: str, repo: Path, *, slice_entry: Mapping[str, Any], config: MilestoneConfig, failure_packet_path: Path | None, failure_packet_data: Mapping[str, Any] | None) -> list[Section]:
    if name == "cmake_neighbors":
        return [Section("Extractor: CMake neighbors", extractor_cmake_neighbors(repo))]
    if name == "minimal_thrend_input":
        return [Section("Extractor: minimal_thrend input.mlir", extractor_minimal_input(repo), "compiler/test/CodeGen/VC4/Hardware/Run/minimal_thrend/input.mlir")]
    if name == "minimal_thrend_reference_qasm":
        return [Section("Extractor: minimal_thrend reference qasm", extractor_minimal_qasm(repo), "compiler/test/CodeGen/VC4/Hardware/Run/minimal_thrend/reference/minimal_thrend.qasm")]
    if name == "existing_codegen_files":
        return [Section("Extractor: existing codegen files", extractor_existing_codegen_files(repo))]
    if name == "git_diff_summary":
        return [Section("Extractor: git diff summary", extractor_git_diff_summary(repo))]
    if name == "changed_files_excerpt":
        return [Section("Extractor: changed files excerpt", extractor_changed_files_excerpt(repo, slice_entry, config))]
    # Static summaries.
    return [Section(f"Extractor: {name}", static_summary(name))]


def load_failure_packet_for_context(repo: Path, failure_packet_path: Path | None) -> tuple[Section | None, Mapping[str, Any] | None]:
    if not failure_packet_path:
        return None, None
    body, data = extractor_failure_packet(repo, failure_packet_path)
    return Section("Failure packet", body, str(failure_packet_path)), data


def build_sections(
    *,
    repo: Path,
    config: MilestoneConfig,
    slice_entry: Mapping[str, Any],
    profile: Mapping[str, Any],
    mode: str,
    failure_packet_path: Path | None,
) -> list[Section]:
    sections: list[Section] = []
    failure_section, failure_packet_data = load_failure_packet_for_context(repo, failure_packet_path)

    sections.append(Section("Context pack metadata", fenced(json.dumps({
        "generated_at_utc": utc_now(),
        "milestone": config.milestone,
        "slice_id": slice_entry.get("id"),
        "slice_title": slice_entry.get("title"),
        "mode": mode,
        "context_profile": slice_entry.get("context_profile"),
        "profile_purpose": profile.get("purpose"),
        "repo": str(repo),
    }, indent=2, sort_keys=True), language="json")))
    sections.append(Section("Active slice contract", format_slice(slice_entry, config)))

    defaults = config.context_profiles.get("defaults", {}) if isinstance(config.context_profiles, dict) else {}
    for item in defaults.get("always_include", []):
        kind = item.get("kind")
        if kind == "file":
            sections.append(include_file(repo, item))
        elif kind == "worklist_slice":
            # Already included as a normalized section, but keep an explicit marker.
            sections.append(Section("Default include: worklist slice", "The active slice contract above is generated from `pro_scripts/vc4_codegen_m1_worklist.json`."))
        elif kind == "output_contract":
            sections.append(include_file(repo, item))
        elif kind == "slice_contract":
            sections.append(include_file(repo, item))
        else:
            sections.append(Section("Unknown default include", f"Unknown default include kind `{kind}`: {item}"))

    if mode in {"failure", "diagnosis"}:
        if failure_section:
            sections.append(failure_section)
        for item in defaults.get("failure_includes", []):
            kind = item.get("kind")
            if kind == "failure_packet":
                if not failure_section:
                    sections.append(Section("Failure packet", "No failure packet was provided."))
            elif kind == "git_diff_summary":
                sections.append(Section("Failure include: git diff summary", extractor_git_diff_summary(repo)))
            elif kind == "changed_files_excerpt":
                sections.append(Section("Failure include: changed files excerpt", extractor_changed_files_excerpt(repo, slice_entry, config)))
            else:
                sections.append(Section("Unknown failure include", f"Unknown failure include kind `{kind}`: {item}"))

    for item in profile.get("include", []):
        kind = item.get("kind")
        if kind == "file":
            sections.append(include_file(repo, item))
        elif kind == "file_excerpt":
            sections.append(include_file_excerpt(repo, item))
        elif kind == "tests":
            sections.extend(include_tests(repo, item))
        elif kind == "fixture":
            sections.append(include_file(repo, item, default_limit=DEFAULT_TEST_FILE_CHAR_LIMIT))
        elif kind == "fixture_tree":
            sections.append(include_fixture_tree(repo, item))
        elif kind == "repo_tree_excerpt":
            roots = [str(x) for x in item.get("roots", [])]
            sections.append(Section("Repository tree excerpt", repo_tree_excerpt(repo, roots), ", ".join(roots)))
        elif kind == "failure_packet":
            if failure_section:
                sections.append(failure_section)
            else:
                sections.append(Section("Failure packet", "No failure packet was provided."))
        elif kind == "generated_artifacts":
            sections.extend(include_generated_artifacts(repo, item, slice_entry, failure_packet_data))
        else:
            sections.append(Section("Unknown include", f"Unknown include kind `{kind}`: {item}"))

    for extractor in profile.get("extractors", []):
        sections.extend(run_extractor(str(extractor), repo, slice_entry=slice_entry, config=config, failure_packet_path=failure_packet_path, failure_packet_data=failure_packet_data))

    sections.append(Section("Context pack usage reminder", """
Use this context only for the active slice. Do not broaden scope. Do not change reference bundles, expected.json, catalog.json, or the GPT web driver. For implementation/failure prompts with a downloadable bundle contract, emit only the exact downloadable bundle artifacts requested by the current prompt; never copy artifact filenames from prior failure packets.
""".strip()))
    return sections



def render_context_pack(
    *,
    repo: Path,
    config: MilestoneConfig,
    slice_id: str,
    mode: str = "initial",
    failure_packet: Path | None = None,
    max_chars_override: int | None = None,
    allow_large_context: bool = False,
    metadata_out: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    slice_entry = config.get_slice(slice_id)
    profile_name = str(slice_entry.get("context_profile", ""))
    profiles = config.context_profiles.get("profiles", {})
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise DriverError(f"slice {slice_id} references missing context profile {profile_name!r}")
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise DriverError(f"context profile {profile_name!r} must be an object")

    defaults = config.context_profiles.get("defaults", {}) if isinstance(config.context_profiles, dict) else {}
    requested_max_chars = parse_optional_int(
        max_chars_override
        if max_chars_override is not None
        else profile.get("max_chars", defaults.get("max_chars", 0)),
        default=0,
    )

    sections = build_sections(repo=repo, config=config, slice_entry=slice_entry, profile=profile, mode=mode, failure_packet_path=failure_packet)
    rendered_sections: list[str] = []
    included: list[dict[str, Any]] = []
    header = f"# VC4 Codegen Context Pack (NO TRUNCATION)\n\nSlice: `{slice_id}` — {slice_entry.get('title')}\n\nProfile: `{profile_name}`\n\nMode: `{mode}`\n\nThis context pack intentionally ignores profile max_chars and includes selected files/logs in full.\n"
    rendered_sections.append(header)
    total = len(header)
    for section in sections:
        text = section.render()
        rendered_sections.append(text)
        total += len(text)
        included.append({"title": section.title, "source": section.source, "chars": len(text), "truncated_to_fit": False, "omitted_budget": False})
    pack = "".join(rendered_sections).rstrip() + "\n"
    metadata = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "slice_id": slice_id,
        "profile": profile_name,
        "mode": mode,
        "chars": len(pack),
        "requested_max_chars_ignored": requested_max_chars,
        "max_chars": None,
        "allow_large_context": True,
        "truncation_policy": "disabled; full selected context included",
        "sections": included,
    }
    if metadata_out:
        write_json_file(metadata_out, metadata)
    return pack, metadata
def cmd_build(args: argparse.Namespace) -> int:
    repo = find_repo_root(args.repo)
    config = MilestoneConfig.load(repo, worklist_path=args.worklist, context_profiles_path=args.context_profiles)
    failure_packet = Path(args.failure_packet) if args.failure_packet else None
    meta_out = Path(args.metadata_out) if args.metadata_out else None
    text, metadata = render_context_pack(
        repo=repo,
        config=config,
        slice_id=args.slice,
        mode=args.mode,
        failure_packet=failure_packet,
        max_chars_override=args.max_chars,
        allow_large_context=args.allow_large_context,
        metadata_out=meta_out,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(json.dumps({"ok": True, "out": str(out), "chars": len(text), "metadata_out": str(meta_out) if meta_out else ""}, indent=2, sort_keys=True))
    else:
        print(text, end="")
    return 0


def cmd_list_profiles(args: argparse.Namespace) -> int:
    repo = find_repo_root(args.repo)
    config = MilestoneConfig.load(repo, worklist_path=args.worklist, context_profiles_path=args.context_profiles)
    profiles = config.context_profiles.get("profiles", {})
    for name, profile in profiles.items():
        print(f"{name}: {profile.get('purpose', '')}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--worklist", default="pro_scripts/vc4_codegen_m1_worklist.json")
    parser.add_argument("--context-profiles", default="pro_scripts/vc4_codegen_m1_context_profiles.json")
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build", help="build a context pack")
    p_build.add_argument("--slice", required=True)
    p_build.add_argument("--mode", choices=["initial", "failure", "diagnosis"], default="initial")
    p_build.add_argument("--failure-packet", default="")
    p_build.add_argument("--out", default="")
    p_build.add_argument("--metadata-out", default="")
    p_build.add_argument("--max-chars", type=int, default=0)
    p_build.add_argument("--allow-large-context", action="store_true")
    p_build.set_defaults(func=cmd_build)

    p_profiles = sub.add_parser("list-profiles")
    p_profiles.set_defaults(func=cmd_list_profiles)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    subcommands = {"build", "list-profiles"}
    # Backward-compatible shorthand: older callers used
    #   vc4_codegen_context_pack.py --repo ... --slice ... --out ...
    # argparse would otherwise reject --slice before we reach the legacy branch.
    if "--slice" in raw_argv and not any(arg in subcommands for arg in raw_argv):
        prefix: list[str] = []
        rest = list(raw_argv)
        for opt in ("--repo", "--worklist", "--context-profiles"):
            if opt in rest:
                i = rest.index(opt)
                if i + 1 < len(rest):
                    prefix.extend([rest[i], rest[i + 1]])
                    del rest[i : i + 2]
        raw_argv = prefix + ["build"] + rest
    args = parser.parse_args(raw_argv)
    if not args.cmd:
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except DriverError as exc:
        print(f"[vc4-context] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
