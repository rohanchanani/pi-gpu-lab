#!/usr/bin/env python3
"""Audit that vc4value remains a tiny two-op launch/policy dialect."""

import argparse
import json
import re
import sys
from pathlib import Path


EXPECTED_ALLOWED_OPS = ["program_id", "num_programs"]
FORBIDDEN_TOKENS = {
    "load",
    "store",
    "tile",
    "vpm",
    "tmu",
    "vdr",
    "vdw",
    "fragment",
    "lane_id",
    "warp_id",
    "thread_id",
    "block_id",
    "barrier",
    "semaphore",
    "qpu",
}
REQUIRED_FEATURES = {
    "vc4value.program_id": {
        "status": "accepted_static_syntax",
        "result_type": "index",
        "axis_values": [0, 1, 2],
    },
    "vc4value.num_programs": {
        "status": "accepted_static_syntax",
        "result_type": "index",
        "axis_values": [0, 1, 2],
    },
    "vc4value.function_metadata_conventions": {
        "status": "documented_convention_phase3_verifier_pending",
    },
    "vc4value.memory_ops": {
        "status": "deterministic_reject_scope_forbidden",
    },
    "vc4value.tile_ops": {
        "status": "deterministic_reject_scope_forbidden",
    },
    "vc4value.fragment_ops": {
        "status": "deterministic_reject_scope_forbidden",
    },
    "vc4value.hardware_path_ops": {
        "status": "deterministic_reject_scope_forbidden",
    },
    "vc4value.lane_id": {
        "status": "deterministic_reject_use_vector_step",
    },
    "vc4value.physical_qpu_identity": {
        "status": "deterministic_reject_not_value_layer_semantics",
    },
    "vc4value.lowering": {
        "status": "not_in_phase2",
    },
}


def fail(message: str) -> None:
    print(f"FAIL VC4Value tiny surface audit: {message}", file=sys.stderr)
    sys.exit(1)


def load_matrix(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        fail(f"could not load matrix {path}: {exc}")


def require_matrix(matrix: dict) -> None:
    if matrix.get("generated_by") != "Phase2e_vc4value_tiny_surface_lock":
        fail("matrix generated_by mismatch")
    if matrix.get("phase") != "Phase2":
        fail("matrix phase must be Phase2")
    if matrix.get("dialect") != "vc4value":
        fail("matrix dialect must be vc4value")
    if matrix.get("allowed_ops") != EXPECTED_ALLOWED_OPS:
        fail(f"allowed_ops must be exactly {EXPECTED_ALLOWED_OPS}")
    if matrix.get("unknown_ops_allowed") is not False:
        fail("unknown_ops_allowed must be false")
    if matrix.get("ready_for_phase3_value_surface_verifier") is not True:
        fail("ready_for_phase3_value_surface_verifier must be true")
    if matrix.get("ready_for_value_to_vc4kernel_lowering") is not False:
        fail("ready_for_value_to_vc4kernel_lowering must be false")
    if matrix.get("ready_for_triton") is not False:
        fail("ready_for_triton must be false")

    rows = {row.get("feature_id"): row for row in matrix.get("features", [])}
    for feature_id, requirements in REQUIRED_FEATURES.items():
        row = rows.get(feature_id)
        if row is None:
            fail(f"missing matrix feature row {feature_id}")
        for key, expected in requirements.items():
            if row.get(key) != expected:
                fail(f"{feature_id} {key} must be {expected!r}; got {row.get(key)!r}")

    hardware_path = rows["vc4value.hardware_path_ops"]
    examples = set(hardware_path.get("forbidden_examples", []))
    for example in ["tmu", "vdr", "vdw", "vpm"]:
        if example not in examples:
            fail(f"hardware_path_ops missing forbidden example {example}")


def extract_ops(ops_td: Path) -> list[tuple[str, str]]:
    text = ops_td.read_text()
    pattern = re.compile(
        r"def\s+VC4Value_([A-Za-z0-9_]+)Op\s*:\s*"
        r"VC4Value_[A-Za-z0-9_]+Op<\"([^\"]+)\""
    )
    return pattern.findall(text)


def require_ops(repo_root: Path) -> None:
    ops_td = repo_root / "compiler/include/vc4/Dialect/VC4Value/IR/VC4ValueOps.td"
    ops = extract_ops(ops_td)
    mnemonics = [mnemonic for _, mnemonic in ops]
    if mnemonics != EXPECTED_ALLOWED_OPS:
        fail(f"VC4Value op mnemonics must be exactly {EXPECTED_ALLOWED_OPS}; got {mnemonics}")

    for def_name, mnemonic in ops:
        checks = [mnemonic, def_name.lower()]
        for token in FORBIDDEN_TOKENS:
            if any(token in check for check in checks):
                fail(f"forbidden token {token!r} found in op {def_name}/{mnemonic}")


def require_no_unknown_ops(repo_root: Path) -> None:
    dialect_cpp = repo_root / "compiler/lib/Dialect/VC4Value/IR/VC4ValueDialect.cpp"
    if "allowUnknownOperations" in dialect_cpp.read_text():
        fail("VC4ValueDialect.cpp must not call allowUnknownOperations")


def require_no_lower_half_headers(repo_root: Path) -> None:
    roots = [
        repo_root / "compiler/include/vc4/Dialect/VC4Value",
        repo_root / "compiler/lib/Dialect/VC4Value",
    ]
    forbidden_patterns = [
        "vc4/Dialect/VC4Kernel/",
        "vc4/Dialect/SSAVC4/",
        "vc4/Dialect/VC4/",
        "vc4/Target/",
        "vc4/Conversion/",
    ]
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            for line_no, line in enumerate(path.read_text().splitlines(), 1):
                if not line.lstrip().startswith("#include"):
                    continue
                for pattern in forbidden_patterns:
                    if pattern in line:
                        rel = path.relative_to(repo_root)
                        fail(f"lower-half include {pattern} in {rel}:{line_no}")


def require_no_conversion_mentions(repo_root: Path) -> None:
    conversion_root = repo_root / "compiler/lib/Conversion"
    if not conversion_root.exists():
        return
    for path in conversion_root.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        if "VC4Value" in text or "vc4value" in text:
            rel = path.relative_to(repo_root)
            fail(f"conversion file mentions VC4Value in Phase 2: {rel}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    if args.mode != "phase2-lock":
        fail(f"unsupported mode {args.mode!r}")

    repo_root = Path(args.repo_root).resolve()
    matrix = load_matrix(Path(args.matrix))
    require_matrix(matrix)
    require_ops(repo_root)
    require_no_unknown_ops(repo_root)
    require_no_lower_half_headers(repo_root)
    require_no_conversion_mentions(repo_root)

    print("PASS VC4Value tiny surface audit: mode=phase2-lock ops=2 unknown_ops_allowed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
