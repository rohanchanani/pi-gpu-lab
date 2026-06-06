#!/usr/bin/env python3
"""Static P0 audit for the VC4Kernel Surface v2 lock."""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ACTIVE_ROOTS = [
    Path("compiler/include/vc4/Dialect/VC4Kernel"),
    Path("compiler/lib/Dialect/VC4Kernel"),
    Path("compiler/lib/Conversion/VC4KernelToSSAVC4"),
    Path("compiler/test/Dialect/VC4Kernel"),
    Path("compiler/test/Conversion/VC4KernelToSSAVC4"),
    Path("compiler/test/CodeGen/VC4Kernel/Hardware/Run"),
]

DIRECT_PATH_ROOTS = [
    Path("compiler/include"),
    Path("compiler/lib"),
    Path("compiler/test"),
]

DIRECT_PATH_TERMS = [
    "VC4KernelToVC4",
    "ConvertVC4KernelToVC4",
    "convert-vc4kernel-to-vc4",
    "vc4kernel-to-vc4",
]

VC4TILE_TERMS = [
    "VC4Tile",
    "vc4tile",
    "VC4_Tile",
    "vc4_tile",
]

FORBIDDEN_TILE_DSL_OPS = [
    "tile_broadcast",
    "tile_dot",
    "tile_matmul",
    "tile_contract",
    "fragment_contract",
    "tile_load",
    "tile_store",
    "copy_tile",
]

SPECIAL_CASE_MATRIX = {
    "fragment_add": ("p1_remove_fragment_add", "P1"),
    "fragment_sub": ("p1_remove_fragment_sub", "P1"),
    "fragment_mul": ("p1_remove_fragment_mul", "P1"),
    "fragment_shl": ("p1_remove_fragment_shl", "P1"),
}

SPECIAL_CASE_ALLOWED_STATUSES = {
    "migration_target",
    "migrated_pending_deletion",
}

PRODUCER_OR_LOWER_HALF_OP_PREFIXES = [
    "vector.",
    "memref.",
    "scf.",
    "linalg.",
    "gpu.",
    "tt.",
    "triton.",
    "ssavc4.",
    "vc4.",
]

TEXT_SUFFIXES = {
    ".td",
    ".h",
    ".cpp",
    ".c",
    ".cc",
    ".mlir",
    ".test",
    ".py",
    ".md",
    ".json",
    ".txt",
}


def fail(message):
    print(f"FAIL VC4Kernel Surface v2 audit: {message}", file=sys.stderr)
    raise SystemExit(1)


def rel(path, root):
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def iter_text_files(root, base):
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = set(path.relative_to(base).parts)
        if relative_parts & {".git", "build", ".vc4_auto"}:
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        yield path


def read_text(path):
    return path.read_text(errors="replace")


def is_support_audit_text(path):
    return "Support" in path.parts and path.suffix == ".py"


def is_negative_test(path):
    name = path.name
    return (
        name.startswith("invalid-")
        or name.startswith("reject-")
        or "invalid" in name
        or "reject" in name
    )


def run_matrix_checker(repo_root, matrix_path):
    checker = (
        repo_root
        / "compiler/test/Dialect/VC4Kernel/Support/check_vc4kernel_surface_v2_matrix.py"
    )
    command = [sys.executable, str(checker), str(matrix_path)]
    result = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        fail(f"support matrix checker failed: {output}")
    return result.stdout.strip().splitlines()


def load_matrix(matrix_path):
    try:
        return json.loads(matrix_path.read_text())
    except json.JSONDecodeError as error:
        fail(f"invalid support matrix JSON: {error}")


def feature_by_id(matrix):
    features = matrix.get("features")
    if not isinstance(features, list):
        fail("support matrix features must be a list")
    return {feature.get("id"): feature for feature in features}


def require_matrix_feature(features, feature_id, phase=None, status=None):
    feature = features.get(feature_id)
    if not feature:
        fail(f"support matrix missing required feature {feature_id}")
    if phase and feature.get("phase") != phase:
        fail(f"feature {feature_id} must be phase {phase}")
    if status and feature.get("current_status") != status:
        fail(f"feature {feature_id} must have current_status {status}")
    return feature


def require_matrix_feature_status_in(features, feature_id, phase, statuses):
    feature = require_matrix_feature(features, feature_id, phase)
    status = feature.get("current_status")
    if status not in statuses:
        allowed = ", ".join(sorted(statuses))
        fail(f"feature {feature_id} must have current_status in {{{allowed}}}")
    return feature


def audit_matrix_ownership(matrix, mode):
    features = feature_by_id(matrix)
    required_special_status = (
        "migrated_pending_deletion" if mode == "p1-migrated" else None
    )
    for op_name, (feature_id, phase) in SPECIAL_CASE_MATRIX.items():
        if required_special_status:
            require_matrix_feature(features, feature_id, phase, required_special_status)
        else:
            require_matrix_feature_status_in(
                features, feature_id, phase, SPECIAL_CASE_ALLOWED_STATUSES
            )
    require_matrix_feature(
        features,
        "p4_remove_add_only_reduce_specialness",
        "P4",
        "migration_target",
    )
    require_matrix_feature(
        features,
        "p7_remove_old_tmu_load_signature",
        "P7",
        "migration_target",
    )
    sparse_p8 = require_matrix_feature(
        features,
        "p8_sparse_vdw_store_deterministic_reject",
        "P8",
        "deterministic_reject",
    )
    sparse_deferred = require_matrix_feature(
        features,
        "deferred_sparse_vdw_store_general_masks",
        "DEFERRED_SPARSE_VDW_STORE",
        "deterministic_reject",
    )
    combined_sparse_text = json.dumps([sparse_p8, sparse_deferred]).lower()
    if "deterministic-reject" not in combined_sparse_text:
        fail("sparse VDW matrix entries must state deterministic-reject")
    policies = matrix.get("locked_policies", {})
    fastmath = policies.get("fastmath_opt_in", {})
    if fastmath.get("opt_in_required") is not True:
        fail("fastmath/SFU opt-in policy is missing from support matrix")
    return {
        "special_case_migration_targets": len(SPECIAL_CASE_MATRIX),
        "reduce_migration_targets": 1,
        "tmu_migration_targets": 1,
        "sparse_vdw_reject_entries": 2,
        "fastmath_opt_in": 1,
    }


def audit_direct_paths(repo_root):
    hits = []
    allowed_audit_hits = 0
    for root in DIRECT_PATH_ROOTS:
        for path in iter_text_files(repo_root / root, repo_root):
            text = read_text(path)
            for term in DIRECT_PATH_TERMS:
                if term not in text:
                    continue
                if is_support_audit_text(path):
                    allowed_audit_hits += 1
                    continue
                hits.append((path, term))
    if hits:
        details = "; ".join(f"{rel(path, repo_root)}:{term}" for path, term in hits)
        fail(f"direct VC4KernelToVC4 path text found: {details}")
    return {"direct_path_hits": 0, "allowed_audit_mentions": allowed_audit_hits}


def audit_vc4tile(repo_root):
    hits = []
    allowed_audit_hits = 0
    for root in DIRECT_PATH_ROOTS:
        for path in iter_text_files(repo_root / root, repo_root):
            text = read_text(path)
            for term in VC4TILE_TERMS:
                if term not in text:
                    continue
                if is_support_audit_text(path):
                    allowed_audit_hits += 1
                    continue
                hits.append((path, term))
    if hits:
        details = "; ".join(f"{rel(path, repo_root)}:{term}" for path, term in hits)
        fail(f"VC4Tile source resurrection text found: {details}")
    return {"retired_tile_hits": 0, "allowed_audit_mentions": allowed_audit_hits}


def audit_forbidden_tile_dsl(repo_root):
    hits = []
    negative_test_mentions = 0
    support_mentions = 0
    scanned = 0
    for root in ACTIVE_ROOTS:
        for path in iter_text_files(repo_root / root, repo_root):
            scanned += 1
            text = read_text(path)
            for op_name in FORBIDDEN_TILE_DSL_OPS:
                spellings = [f"vc4kernel.{op_name}", op_name]
                if not any(spelling in text for spelling in spellings):
                    continue
                if is_support_audit_text(path):
                    support_mentions += 1
                    continue
                if is_negative_test(path):
                    negative_test_mentions += 1
                    continue
                hits.append((path, op_name))
    if hits:
        details = "; ".join(f"{rel(path, repo_root)}:{op}" for path, op in hits)
        fail(f"forbidden tile DSL op found in active surface: {details}")
    return {
        "active_files_scanned": scanned,
        "forbidden_tile_dsl_hits": 0,
        "negative_test_mentions": negative_test_mentions,
        "support_mentions": support_mentions,
    }


def audit_fixture_purity(repo_root):
    fixture_root = repo_root / "compiler/test/CodeGen/VC4Kernel/Hardware/Run"
    inputs = sorted(fixture_root.rglob("input.mlir"))
    if not inputs:
        fail("no VC4Kernel hardware fixture input.mlir files found")
    bad = []
    producer_hits = Counter()
    for path in inputs:
        text = read_text(path)
        if "vc4kernel.kernel" not in text:
            bad.append(f"{rel(path, repo_root)}: missing vc4kernel.kernel")
        for prefix in PRODUCER_OR_LOWER_HALF_OP_PREFIXES:
            if prefix in text:
                producer_hits[prefix] += 1
                bad.append(f"{rel(path, repo_root)}: contains {prefix}")
    if bad:
        fail("fixture purity violations: " + "; ".join(bad[:20]))
    return {
        "hardware_inputs": len(inputs),
        "producer_or_lower_half_hits": sum(producer_hits.values()),
    }


def audit_special_case_presence(repo_root, matrix_counts):
    op_root = repo_root / "compiler/include/vc4/Dialect/VC4Kernel/IR"
    text = "\n".join(read_text(path) for path in sorted(op_root.glob("*.td")))
    present = []
    missing = []
    for op_name in SPECIAL_CASE_MATRIX:
        if op_name in text:
            present.append(op_name)
        else:
            missing.append(op_name)
    if missing:
        fail(
            "P0 baseline expected special-case migration ops to be present: "
            + ", ".join(missing)
        )
    if matrix_counts["special_case_migration_targets"] != len(present):
        fail("special-case migration count does not match matrix ownership")
    return {"present_special_case_ops": len(present)}


def audit_no_legacy_user_spellings(repo_root):
    roots = [
        Path("compiler/test/Dialect/VC4Kernel"),
        Path("compiler/test/Conversion/VC4KernelToSSAVC4"),
        Path("compiler/test/CodeGen/VC4Kernel"),
        Path("compiler/docs"),
    ]
    spellings = [
        "vc4kernel." + "fragment_add",
        "vc4kernel." + "fragment_sub",
        "vc4kernel." + "fragment_mul",
        "vc4kernel." + "fragment_shl",
    ]
    hits = []
    scanned = 0
    for root in roots:
        for path in iter_text_files(repo_root / root, repo_root):
            scanned += 1
            text = read_text(path)
            for spelling in spellings:
                if spelling in text:
                    hits.append((path, spelling))
    if hits:
        details = "; ".join(f"{rel(path, repo_root)}:{spelling}" for path, spelling in hits[:20])
        fail(f"P1 migrated mode found legacy user spellings: {details}")
    return {"legacy_user_spelling_hits": 0, "legacy_user_files_scanned": scanned}


def format_counts(counts):
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--matrix",
        default="compiler/docs/vc4kernel_surface_v2_support_matrix.json",
    )
    parser.add_argument("--mode", default="p0-baseline")
    parser.add_argument("--phase-lock", default=None)
    args = parser.parse_args(argv)

    if args.mode not in {"p0-baseline", "p1-migrated"}:
        fail(f"unsupported audit mode: {args.mode}")
    repo_root = Path(args.repo_root).resolve()
    matrix_path = Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = repo_root / matrix_path
    if not matrix_path.is_file():
        fail(f"matrix path does not exist: {matrix_path}")

    matrix_summary = run_matrix_checker(repo_root, matrix_path)
    matrix = load_matrix(matrix_path)
    matrix_counts = audit_matrix_ownership(matrix, args.mode)
    special_case_counts = audit_special_case_presence(repo_root, matrix_counts)
    direct_counts = audit_direct_paths(repo_root)
    vc4tile_counts = audit_vc4tile(repo_root)
    tile_counts = audit_forbidden_tile_dsl(repo_root)
    fixture_counts = audit_fixture_purity(repo_root)
    legacy_user_counts = (
        audit_no_legacy_user_spellings(repo_root)
        if args.mode == "p1-migrated"
        else {}
    )

    migration_summary = {}
    migration_summary.update(matrix_counts)
    migration_summary.update(special_case_counts)

    forbidden_summary = {}
    forbidden_summary.update(direct_counts)
    forbidden_summary.update(vc4tile_counts)
    forbidden_summary.update(tile_counts)

    print(f"PASS VC4Kernel Surface v2 audit: mode={args.mode}")
    if matrix_summary:
        print(f"matrix_checker: {matrix_summary[0]}")
    print(f"migration_targets: {format_counts(migration_summary)}")
    print(f"forbidden_path_scan: {format_counts(forbidden_summary)}")
    print(f"fixture_purity: {format_counts(fixture_counts)}")
    if legacy_user_counts:
        print(f"legacy_user_scan: {format_counts(legacy_user_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
