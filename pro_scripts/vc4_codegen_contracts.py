#!/usr/bin/env python3
"""Executable repository contracts for VC4 codegen Milestone 1 automation."""
from __future__ import annotations

import json, os, py_compile, re, shlex, shutil, subprocess, sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from vc4_codegen_state import DriverError, git_changed_paths, normalize_relpath, relpath
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vc4_codegen_state import DriverError, git_changed_paths, normalize_relpath, relpath  # type: ignore

RUN_LINE_RE = re.compile(r"^\s*(?://|#)\s*RUN:\s*(.*)$")
PERCENT_TOKEN_RE = re.compile(r"%(?!\{)([A-Za-z_][A-Za-z0-9_-]*)")
TOOL_SUBST_RE = re.compile(r"ToolSubst\(\s*['\"](%[A-Za-z_][A-Za-z0-9_.-]*)['\"]")
SUBST_APPEND_RE = re.compile(r"(?:config\.)?substitutions\.append\(\s*\(\s*['\"](%[A-Za-z_][A-Za-z0-9_.-]*)['\"]")
ADD_TOOL_RE = re.compile(r"\b(?:add_mlir_tool|add_llvm_tool|add_llvm_executable|add_executable)\s*\(\s*([A-Za-z0-9_.+-]+)")

LIT_BUILTINS = {"%s","%S","%p","%T","%t","%basename_t","%/s","%/S","%/p","%/T","%/t","%{pathsep}","%{python}","%python","%shlibext","%exeext"}
PLUMBING_COMMANDS = {"cat","cd","cmp","cp","diff","echo","env","false","FileCheck","grep","head","mkdir","mv","not","printf","pwd","rm","sed","sh","bash","test","touch","true","wc","xargs"}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _iter_files(repo: Path, rels: Iterable[str]):
    for raw in sorted(set(rels)):
        try: rel = normalize_relpath(raw)
        except DriverError: continue
        path = repo / rel
        if path.exists() and path.is_file():
            yield rel, path


def extract_run_lines(text: str) -> list[str]:
    out: list[str] = []
    pending: str | None = None
    for line in text.splitlines():
        m = RUN_LINE_RE.match(line)
        if m:
            if pending is not None:
                out.append(pending)
            body = m.group(1).rstrip()
            if body.endswith("\\"):
                pending = body[:-1].rstrip() + " "
            else:
                out.append(body)
        elif pending is not None:
            body = line.strip()
            if body.endswith("\\"):
                pending += body[:-1].rstrip() + " "
            else:
                pending += body
                out.append(pending)
                pending = None
    if pending is not None:
        out.append(pending)
    return out


def discover_lit_substitutions(repo: Path) -> dict[str, Any]:
    subs = set(LIT_BUILTINS)
    sources: dict[str, list[str]] = {}
    root = repo / "compiler/test"
    configs = sorted(root.glob("**/lit*.cfg*")) if root.exists() else []
    for path in configs:
        if not path.is_file():
            continue
        rel = relpath(repo, path)
        try: text = _read(path)
        except OSError: continue
        for rx in (TOOL_SUBST_RE, SUBST_APPEND_RE):
            for m in rx.finditer(text):
                tok = m.group(1)
                subs.add(tok)
                sources.setdefault(tok, []).append(rel)
    return {"substitutions": sorted(subs), "sources": sources, "config_files": [relpath(repo, p) for p in configs if p.is_file()]}


def discover_cmake_tool_targets(repo: Path) -> dict[str, Any]:
    targets: set[str] = set(); sources: dict[str, list[str]] = {}
    root = repo / "compiler"
    for path in sorted(root.glob("**/CMakeLists.txt")) if root.exists() else []:
        try: text = _read(path)
        except OSError: continue
        rel = relpath(repo, path)
        for m in ADD_TOOL_RE.finditer(text):
            target = m.group(1)
            targets.add(target)
            sources.setdefault(target, []).append(rel)
    return {"targets": sorted(targets), "sources": sources}


def discover_build_bin_tools(repo: Path) -> list[str]:
    d = repo / "compiler/build/bin"
    if not d.exists(): return []
    return sorted(p.name for p in d.iterdir() if p.is_file() and os.access(p, os.X_OK))


def _split_segments(line: str) -> list[str]:
    return [p.strip() for p in re.split(r"\s*(?:\|\||&&|\||;)\s*", line) if p.strip()]


def extract_shell_commands(run_line: str) -> list[str]:
    out: list[str] = []
    for seg in _split_segments(run_line):
        try: toks = shlex.split(seg, posix=True)
        except ValueError: continue
        if not toks: continue
        i = 0
        while i < len(toks) and "=" in toks[i] and not toks[i].startswith("-"):
            i += 1
        while i < len(toks) and toks[i] in {"not", "env"}:
            i += 1
            while i < len(toks) and "=" in toks[i] and not toks[i].startswith("-"):
                i += 1
        if i < len(toks): out.append(toks[i])
    return out


def build_repo_capability_snapshot(repo: Path, slice_entry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    lit = discover_lit_substitutions(repo); cmake = discover_cmake_tool_targets(repo)
    path_tools = sorted(x for x in ["vc4-opt","vc4-codegen","vc4asm","FileCheck","llvm-lit","lit","ninja","cmake"] if shutil.which(x))
    return {
        "schema_version": 1,
        "slice_id": slice_entry.get("id") if slice_entry else None,
        "declared_gates": list(slice_entry.get("gates", [])) if slice_entry else [],
        "lit_substitutions": lit["substitutions"],
        "lit_substitution_sources": lit["sources"],
        "lit_config_files": lit["config_files"],
        "cmake_tool_targets": cmake["targets"],
        "cmake_tool_target_sources": cmake["sources"],
        "build_bin_tools": discover_build_bin_tools(repo),
        "path_tools_detected": path_tools,
        "contracts": [
            "Changed lit RUN lines must not use unresolved custom %tokens.",
            "Bare project tools in changed RUN lines must resolve through PATH, compiler/build/bin, or CMake target discovery.",
            "Changed lit Python config must parse before the full lit suite runs.",
        ],
    }


def _is_test_file(rel: str) -> bool:
    return rel.startswith("compiler/test/") and Path(rel).suffix in {".mlir", ".ll", ".td", ".txt", ".test"}


def _is_lit_config(rel: str) -> bool:
    return rel.startswith("compiler/test/") and Path(rel).name in {"lit.cfg.py", "lit.local.cfg"}


def _percent_tokens(line: str) -> list[str]:
    return ["%" + m.group(1) for m in PERCENT_TOKEN_RE.finditer(line)]


def _command_resolves(cmd: str, repo: Path, snapshot: Mapping[str, Any]) -> bool:
    if not cmd or cmd.startswith("-") or cmd.startswith("%") or "/" in cmd or cmd in PLUMBING_COMMANDS:
        return True
    return bool(shutil.which(cmd) or cmd in set(snapshot.get("build_bin_tools", [])) or cmd in set(snapshot.get("cmake_tool_targets", [])) or (repo / "compiler/build/bin" / cmd).exists())


def validate_patch_invariants(repo: Path, slice_entry: Mapping[str, Any], changed_paths: Sequence[str] | None = None) -> dict[str, Any]:
    repo = repo.resolve()
    changed = [normalize_relpath(p) for p in (changed_paths if changed_paths is not None else git_changed_paths(repo, include_untracked=True))]
    snapshot = build_repo_capability_snapshot(repo, slice_entry)
    substitutions = set(str(x) for x in snapshot.get("lit_substitutions", []))
    errors: list[dict[str, Any]] = []; warnings: list[dict[str, Any]] = []

    for rel, path in _iter_files(repo, changed):
        if _is_lit_config(rel):
            try: py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                errors.append({"category": "lit_config_syntax", "path": rel, "message": f"changed lit config does not parse as Python: {exc.msg}"})
        if not _is_test_file(rel):
            continue
        for idx, line in enumerate(extract_run_lines(_read(path)), start=1):
            for tok in sorted(set(_percent_tokens(line))):
                if tok not in substitutions:
                    errors.append({"category": "lit_tool_resolution", "path": rel, "run_line_index": idx, "token": tok, "run_line": line, "message": f"RUN line uses custom lit substitution {tok!r}, but no lit config defines it"})
            for cmd in extract_shell_commands(line):
                if not _command_resolves(cmd, repo, snapshot):
                    errors.append({"category": "tool_resolution", "path": rel, "run_line_index": idx, "command": cmd, "run_line": line, "message": f"RUN line invokes bare command {cmd!r}, but it is not resolvable"})

    cmake_targets = set(str(x) for x in snapshot.get("cmake_tool_targets", [])) | set(str(x) for x in snapshot.get("build_bin_tools", []))
    build_ninja = repo / "compiler/build/build.ninja"
    ninja_text = _read(build_ninja) if build_ninja.exists() else ""
    for gate in [str(g) for g in slice_entry.get("gates", [])]:
        if gate.startswith("build:"):
            target = gate.split(":", 1)[1]
            if target == "check-vc4" or target in cmake_targets or re.search(rf"(?:^|\s){re.escape(target)}(?:\s|$)", ninja_text):
                continue
            errors.append({"category": "cmake_or_target", "gate": gate, "target": target, "message": f"declared build gate {gate!r} has no discoverable CMake/build target after the candidate patch"})

    return {"schema_version": 1, "ok": not errors, "category": (errors[0]["category"] if errors else "ok"), "slice_id": slice_entry.get("id"), "changed_paths": changed, "errors": errors, "warnings": warnings, "repo_capabilities": snapshot}


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

# Backward-compatible names used by the hardened prompt renderer/autorunner.
def build_repo_capabilities(repo: Path, slice_entry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return build_repo_capability_snapshot(repo, slice_entry)


def render_capabilities_markdown(snapshot: Mapping[str, Any]) -> str:
    lines = [
        "## Executable repo/test/tool contract snapshot",
        "",
        "This snapshot is generated from the repository before each prompt. Treat it as executable contract context.",
        "",
        "### Lit substitutions",
    ]
    for tok in snapshot.get("lit_substitutions", []) or ["<none>"]:
        lines.append(f"- `{tok}`")
    lines += ["", "### Lit config files"]
    for cfg in snapshot.get("lit_config_files", []) or ["<none>"]:
        lines.append(f"- `{cfg}`")
    lines += ["", "### CMake/compiler tool targets"]
    for target in snapshot.get("cmake_tool_targets", []) or ["<none>"]:
        lines.append(f"- `{target}`")
    lines += ["", "### Already-built compiler/build/bin tools"]
    for tool in snapshot.get("build_bin_tools", []) or ["<none>"]:
        lines.append(f"- `{tool}`")
    lines += ["", "### Contract notes"]
    for note in snapshot.get("contracts", []) or []:
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"


def render_repo_capabilities_markdown(repo: Path) -> str:
    return render_capabilities_markdown(build_repo_capabilities(repo))

