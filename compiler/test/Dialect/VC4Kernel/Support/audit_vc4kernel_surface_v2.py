#!/usr/bin/env python3
"""Static P0 audit for the VC4Kernel Surface v2 lock."""

import argparse
import json
import re
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
    "removed_in_p1",
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

P3_I32_CMP_PREDICATES = {
    "eq",
    "ne",
    "ult",
    "ule",
    "ugt",
    "uge",
    "slt",
    "sle",
    "sgt",
    "sge",
}

P3_F32_ORDERED_CMP_PREDICATES = {
    "oeq",
    "one",
    "olt",
    "ole",
    "ogt",
    "oge",
}

P3_UNORDERED_CMP_PREDICATES = {
    "uno",
    "ueq",
    "une",
    "ult_unordered",
    "ule_unordered",
    "ugt_unordered",
    "uge_unordered",
}

P3_POST_PHASES = {
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "P9",
    "P10",
    "P11",
    "P12",
    "P13",
}

P3_POST_PHASE_ALLOWED_STATUSES = {
    "planned",
    "migration_target",
    "deterministic_reject",
}

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
    required_special_status = None
    if mode == "p1-migrated":
        required_special_status = "migrated_pending_deletion"
    if mode in {
        "p1-general-alu-lock",
        "p2-bitcast-const-lock",
        "p3-general-cmp-lock",
    }:
        required_special_status = "removed_in_p1"
    for op_name, (feature_id, phase) in SPECIAL_CASE_MATRIX.items():
        if required_special_status:
            require_matrix_feature(features, feature_id, phase, required_special_status)
        else:
            require_matrix_feature_status_in(
                features, feature_id, phase, SPECIAL_CASE_ALLOWED_STATUSES
            )
    if mode in {
        "p1-general-alu-lock",
        "p2-bitcast-const-lock",
        "p3-general-cmp-lock",
    }:
        require_matrix_feature(features, "p1_general_fragment_add_alu", "P1", "accepted")
        require_matrix_feature(features, "p1_general_fragment_mul_alu", "P1", "accepted")
    p2_status_entries = 0
    if mode == "p2-bitcast-const-lock":
        allowed_p2_statuses = {"accepted", "deterministic_reject"}
        for feature in features.values():
            if feature.get("phase") != "P2":
                continue
            p2_status_entries += 1
            status = feature.get("current_status")
            if status not in allowed_p2_statuses:
                fail(
                    f"P2 feature {feature.get('id')} must be accepted or deterministic_reject"
                )
        require_matrix_feature(features, "p2_fragment_bitcast", "P2", "accepted")
        require_matrix_feature(
            features,
            "p2_fragment_const_splat_zero_allones_lane_affine",
            "P2",
            "accepted",
        )
    p3_status_entries = 0
    p3_post_phase_planned_entries = 0
    if mode == "p3-general-cmp-lock":
        accepted_or_pending = {
            "accepted",
            "hardware_proven_pending_final_acceptance",
        }
        signed_i32 = require_matrix_feature_status_in(
            features,
            "p3_fragment_cmp_signed_i32",
            "P3",
            accepted_or_pending,
        )
        ordered_f32 = require_matrix_feature_status_in(
            features,
            "p3_fragment_cmp_ordered_f32",
            "P3",
            accepted_or_pending,
        )
        p3_status_entries = 2
        ordered_f32_text = json.dumps(ordered_f32).lower()
        if (
            "finite_only" not in ordered_f32_text
            or "unordered" not in ordered_f32_text
            or "nan" not in ordered_f32_text
            or "deterministic" not in ordered_f32_text
        ):
            fail(
                "P3 f32 comparison matrix entry must document finite_only and "
                "unordered/NaN deterministic rejection"
            )
        signed_i32_text = json.dumps(signed_i32).lower()
        if "signed" not in signed_i32_text or "i32" not in signed_i32_text:
            fail("P3 signed i32 comparison matrix entry must document signed i32")
        for feature in features.values():
            phase = feature.get("phase")
            if phase not in P3_POST_PHASES:
                continue
            p3_post_phase_planned_entries += 1
            status = feature.get("current_status")
            if status not in P3_POST_PHASE_ALLOWED_STATUSES:
                fail(
                    f"P3 lock expected post-P3 feature {feature.get('id')} "
                    f"to remain planned/migration_target/deterministic_reject"
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
        "p2_locked_status_entries": p2_status_entries,
        "p3_locked_status_entries": p3_status_entries,
        "p3_post_phase_planned_entries": p3_post_phase_planned_entries,
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


def audit_special_case_presence(repo_root, matrix_counts, mode):
    op_root = repo_root / "compiler/include/vc4/Dialect/VC4Kernel/IR"
    text = "\n".join(read_text(path) for path in sorted(op_root.glob("*.td")))
    present = []
    missing = []
    for op_name in SPECIAL_CASE_MATRIX:
        if op_name in text:
            present.append(op_name)
        else:
            missing.append(op_name)
    if mode in {
        "p1-general-alu-lock",
        "p2-bitcast-const-lock",
        "p3-general-cmp-lock",
    }:
        if present:
            fail("P1 general ALU lock expected legacy ops to be absent: " + ", ".join(present))
        return {"present_special_case_ops": 0, "removed_special_case_ops": len(missing)}
    if missing and mode == "p0-baseline":
        removed = matrix_counts.get("special_case_removed_in_p1", 0)
        if removed == len(SPECIAL_CASE_MATRIX):
            return {"present_special_case_ops": 0, "removed_special_case_ops": len(missing)}
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


def audit_p1_general_alu_lock(repo_root):
    active_roots = [
        Path("compiler/include/vc4/Dialect/VC4Kernel"),
        Path("compiler/lib/Dialect/VC4Kernel"),
        Path("compiler/lib/Conversion/VC4KernelToSSAVC4"),
        Path("compiler/test/Dialect/VC4Kernel"),
        Path("compiler/test/Conversion/VC4KernelToSSAVC4"),
        Path("compiler/test/CodeGen/VC4Kernel/Hardware/Run"),
    ]
    doc_root = Path("compiler/docs")
    forbidden_terms = [
        "vc4kernel." + "fragment_add",
        "vc4kernel." + "fragment_sub",
        "vc4kernel." + "fragment_mul",
        "vc4kernel." + "fragment_shl",
        "kFragment" + "AddOpName",
        "kFragment" + "SubOpName",
        "kFragment" + "MulOpName",
        "kFragment" + "ShlOpName",
        "Fragment" + "AddOp",
        "Fragment" + "SubOp",
        "Fragment" + "MulOp",
        "Fragment" + "ShlOp",
    ]
    active_hits = []
    scanned = 0
    for root in active_roots:
        for path in iter_text_files(repo_root / root, repo_root):
            scanned += 1
            text = read_text(path)
            for term in forbidden_terms:
                if term in text:
                    active_hits.append((path, term))
    if active_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{term}" for path, term in active_hits[:20]
        )
        fail(f"P1 general ALU lock found removed legacy op text: {details}")

    doc_hits = []
    allowed_doc_hits = 0
    allowed_markers = ("historical", "removed_in_p1", "legacy migration")
    for path in iter_text_files(repo_root / doc_root, repo_root):
        for lineno, line in enumerate(read_text(path).splitlines(), start=1):
            for term in forbidden_terms:
                if term not in line:
                    continue
                if any(marker in line.lower() for marker in allowed_markers):
                    allowed_doc_hits += 1
                    continue
                doc_hits.append((path, lineno, term))
    if doc_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{lineno}:{term}"
            for path, lineno, term in doc_hits[:20]
        )
        fail(f"P1 general ALU lock found non-historical docs legacy text: {details}")
    return {
        "active_files_scanned": scanned,
        "removed_legacy_hits": 0,
        "allowed_historical_doc_hits": allowed_doc_hits,
    }


def count_exact_static_const_splats(repo_root):
    roots = [
        Path("compiler/test/Dialect/VC4Kernel"),
        Path("compiler/test/Conversion/VC4KernelToSSAVC4"),
        Path("compiler/test/CodeGen/VC4Kernel/Hardware/Run"),
        Path("compiler/docs/examples"),
    ]
    const_re = re.compile(
        r"^\s*(%[\w.$-]+)\s*=\s*arith\.constant\s+(.+?)\s*:\s*(i32|f32)\s*$"
    )
    splat_re = re.compile(
        r"^\s*(%[\w.$-]+)\s*=\s*vc4kernel\.splat\s+(%[\w.$-]+)\s*:\s*"
        r"(i32|f32)\s*->\s*(vector<16x(?:i32|f32)>)\s*$"
    )
    hits = []
    scanned = 0
    for root in roots:
        full_root = repo_root / root
        if not full_root.exists():
            continue
        for path in iter_text_files(full_root, repo_root):
            if path.suffix != ".mlir":
                continue
            scanned += 1
            constants = {}
            lines = read_text(path).splitlines()
            for lineno, line in enumerate(lines, start=1):
                match = const_re.match(line)
                if match:
                    constants[match.group(1)] = (match.group(2), match.group(3), lineno)
            for lineno, line in enumerate(lines, start=1):
                match = splat_re.match(line)
                if not match:
                    continue
                source = match.group(2)
                if source in constants and match.group(3) == constants[source][1]:
                    hits.append((path, lineno, source))
    if hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{lineno}:{source}"
            for path, lineno, source in hits[:20]
        )
        fail(f"P2 lock found compile-time arith.constant -> vc4kernel.splat: {details}")
    return {"static_const_splat_hits": 0, "static_const_splat_files_scanned": scanned}


def audit_p2_bitcast_const_lock(repo_root, matrix):
    conversion_path = (
        repo_root
        / "compiler/lib/Conversion/VC4KernelToSSAVC4/VC4KernelToSSAVC4.cpp"
    )
    conversion_text = read_text(conversion_path)
    bitcast_marker = "if (hasName(op, kFragmentBitcastOpName))"
    const_marker = "if (hasName(op, kFragmentConstOpName))"
    bitcast_index = conversion_text.find(bitcast_marker)
    if bitcast_index < 0:
        fail("P2 lock expected fragment_bitcast lowering marker")
    next_index = conversion_text.find(const_marker, bitcast_index + len(bitcast_marker))
    bitcast_block = conversion_text[bitcast_index: next_index if next_index >= 0 else None]
    numeric_hits = sum(
        term in bitcast_block
        for term in [
            "itof",
            "ftoi",
            "VC4Kernel_AddALUOpcode::itof",
            "VC4Kernel_AddALUOpcode::ftoi",
            "kSSAVC4ALUAddOpName",
        ]
    )
    if "kSSAVC4MovOpName" not in bitcast_block:
        fail("P2 lock expected fragment_bitcast lowering to ssavc4.mov")
    if numeric_hits:
        fail("P2 lock found numeric conversion in fragment_bitcast lowering")

    ops_cpp = read_text(
        repo_root / "compiler/lib/Dialect/VC4Kernel/IR/VC4KernelOps.cpp"
    )
    reject_diag = "requires efficient fragment_const materialization"
    finite_diag = "fragment_const f32 splat must be finite in P2"
    if reject_diag not in ops_cpp:
        fail("P2 lock expected fragment_const efficient-materialization diagnostic")
    if finite_diag not in ops_cpp:
        fail("P2 lock expected fragment_const NaN/Inf rejection diagnostic")
    reject_test = (
        repo_root
        / "compiler/test/Dialect/VC4Kernel/fragment_const_reject_arbitrary_dense.mlir"
    )
    if not reject_test.is_file() or reject_diag not in read_text(reject_test):
        fail("P2 lock expected arbitrary dense fragment_const reject test")

    hardware_legacy_hits = []
    legacy_terms = [
        "vc4kernel." + "fragment_add",
        "vc4kernel." + "fragment_sub",
        "vc4kernel." + "fragment_mul",
        "vc4kernel." + "fragment_shl",
    ]
    for path in sorted(
        (repo_root / "compiler/test/CodeGen/VC4Kernel/Hardware/Run").rglob(
            "input.mlir"
        )
    ):
        text = read_text(path)
        for term in legacy_terms:
            if term in text:
                hardware_legacy_hits.append((path, term))
    if hardware_legacy_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{term}"
            for path, term in hardware_legacy_hits[:20]
        )
        fail(f"P2 lock found legacy fragment arithmetic in hardware fixtures: {details}")

    features = feature_by_id(matrix)
    p2_matrix_statuses = Counter()
    for feature in features.values():
        if feature.get("phase") == "P2":
            p2_matrix_statuses[feature.get("current_status")] += 1
    if p2_matrix_statuses.get("accepted", 0) < 2:
        fail("P2 lock expected accepted bitcast and fragment_const matrix entries")

    counts = count_exact_static_const_splats(repo_root)
    counts.update(
        {
            "bitcast_uses_mov": 1,
            "bitcast_numeric_conversion_hits": 0,
            "fragment_const_reject_diagnostic": 1,
            "nan_inf_reject_diagnostic": 1,
            "hardware_legacy_arith_hits": 0,
            "p2_matrix_accepted": p2_matrix_statuses.get("accepted", 0),
            "p2_matrix_deterministic_reject": p2_matrix_statuses.get(
                "deterministic_reject", 0
            ),
        }
    )
    return counts


def collect_fragment_cmp_ops(text):
    cmp_ops = []
    for line in text.splitlines():
        if "vc4kernel.fragment_cmp" not in line:
            continue
        if "// CHECK:" in line:
            continue
        predicate_match = re.search(r"#vc4kernel\.cmp<([^>]+)>", line)
        cmp_ops.append(
            {
                "line": line,
                "predicate": predicate_match.group(1) if predicate_match else None,
                "has_finite_only_policy": "#vc4kernel.fp_cmp_policy<finite_only>" in line,
                "has_f32_type": "vector<16xf32>" in line,
                "has_i32_type": "vector<16xi32>" in line,
            }
        )
    return cmp_ops


def audit_p3_general_cmp_lock(repo_root, matrix):
    features = feature_by_id(matrix)
    require_matrix_feature_status_in(
        features,
        "p3_fragment_cmp_signed_i32",
        "P3",
        {"accepted", "hardware_proven_pending_final_acceptance"},
    )
    require_matrix_feature_status_in(
        features,
        "p3_fragment_cmp_ordered_f32",
        "P3",
        {"accepted", "hardware_proven_pending_final_acceptance"},
    )

    matrix_scalar_claims = []
    for feature in features.values():
        if feature.get("phase") != "P3":
            continue
        text = json.dumps(feature).lower()
        status = feature.get("current_status")
        if "arith.cmpi" in text and status in {
            "accepted",
            "hardware_proven_pending_final_acceptance",
            "implemented_pending_hardware",
            "implemented_hardware_pending_final_acceptance",
        }:
            matrix_scalar_claims.append(feature.get("id"))
    if matrix_scalar_claims:
        fail(
            "P3 lock found scalar arith.cmpi expansion claimed by P3: "
            + ", ".join(matrix_scalar_claims)
        )

    roots = [
        Path("compiler/include/vc4/Dialect/VC4Kernel"),
        Path("compiler/lib/Dialect/VC4Kernel"),
        Path("compiler/lib/Conversion/VC4KernelToSSAVC4"),
        Path("compiler/test/Dialect/VC4Kernel"),
        Path("compiler/test/Conversion/VC4KernelToSSAVC4"),
        Path("compiler/test/CodeGen/VC4Kernel/Hardware/Run"),
        Path("compiler/docs"),
    ]
    active_cmp_uses = 0
    i32_cmp_uses = 0
    f32_finite_cmp_uses = 0
    f32_missing_policy_hits = []
    unordered_active_hits = []
    fp_policy_on_i32_hits = []
    unordered_negative_mentions = 0
    docs_historical_mentions = 0
    files_scanned = 0

    historical_markers = ("historical", "removed", "old")
    for root in roots:
        for path in iter_text_files(repo_root / root, repo_root):
            if is_support_audit_text(path):
                continue
            files_scanned += 1
            text = read_text(path)
            for cmp_op in collect_fragment_cmp_ops(text):
                predicate = cmp_op["predicate"]
                if predicate is None and path.suffix not in {".mlir", ".md"}:
                    continue
                is_negative = is_negative_test(path)
                is_doc = "compiler/docs" in str(rel(path, repo_root))
                if predicate is None and is_doc:
                    docs_historical_mentions += 1
                    continue
                if is_doc and any(
                    marker in cmp_op["line"].lower() for marker in historical_markers
                ):
                    docs_historical_mentions += 1
                    continue
                if predicate in P3_UNORDERED_CMP_PREDICATES:
                    if is_negative:
                        unordered_negative_mentions += 1
                    else:
                        unordered_active_hits.append((path, predicate))
                    continue
                if predicate in P3_F32_ORDERED_CMP_PREDICATES:
                    if not cmp_op["has_finite_only_policy"]:
                        if is_negative:
                            continue
                        f32_missing_policy_hits.append((path, predicate))
                        continue
                    if not cmp_op["has_f32_type"]:
                        if is_negative:
                            continue
                        fail(
                            "P3 lock found f32 comparison predicate without f32 "
                            f"fragment types: {rel(path, repo_root)}:{predicate}"
                        )
                    active_cmp_uses += 1
                    f32_finite_cmp_uses += 1
                    continue
                if predicate in P3_I32_CMP_PREDICATES:
                    if cmp_op["has_finite_only_policy"]:
                        if is_negative:
                            continue
                        fp_policy_on_i32_hits.append((path, predicate))
                        continue
                    if not cmp_op["has_i32_type"]:
                        if is_negative:
                            continue
                        fail(
                            "P3 lock found integer comparison predicate without i32 "
                            f"fragment types: {rel(path, repo_root)}:{predicate}"
                        )
                    active_cmp_uses += 1
                    i32_cmp_uses += 1
                    continue
                if predicate is None and is_negative:
                    continue
                fail(
                    "P3 lock found unclassified fragment_cmp predicate "
                    f"{predicate!r} in {rel(path, repo_root)}"
                )

    if f32_missing_policy_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{predicate}"
            for path, predicate in f32_missing_policy_hits[:20]
        )
        fail(f"P3 lock found f32 fragment_cmp without finite_only policy: {details}")
    if unordered_active_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{predicate}"
            for path, predicate in unordered_active_hits[:20]
        )
        fail(f"P3 lock found active unordered fragment_cmp predicate: {details}")
    if fp_policy_on_i32_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{predicate}"
            for path, predicate in fp_policy_on_i32_hits[:20]
        )
        fail(f"P3 lock found fp_policy on active i32 fragment_cmp: {details}")

    fixture_root = repo_root / "compiler/test/CodeGen/VC4Kernel/Hardware/Run"
    p3_f32_nan_inf_hits = []
    nan_inf_re = re.compile(r"\b(?:nan|inf|infinity|NAN|INF|INFINITY)\b")
    for path in sorted(fixture_root.glob("fragment_cmp_f32*_vc4kernel/*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        text = read_text(path)
        if nan_inf_re.search(text):
            p3_f32_nan_inf_hits.append(path)
    if p3_f32_nan_inf_hits:
        details = "; ".join(
            str(rel(path, repo_root)) for path in p3_f32_nan_inf_hits[:20]
        )
        fail(f"P3 lock found NaN/Inf text in f32 comparison fixture: {details}")

    if i32_cmp_uses == 0 or f32_finite_cmp_uses == 0:
        fail("P3 lock expected both i32 and finite f32 fragment_cmp uses")

    return {
        "active_cmp_uses": active_cmp_uses,
        "cmp_files_scanned": files_scanned,
        "f32_finite_cmp_uses": f32_finite_cmp_uses,
        "f32_missing_policy_hits": 0,
        "fp_policy_on_i32_hits": 0,
        "i32_cmp_uses": i32_cmp_uses,
        "matrix_scalar_cmpi_claims": 0,
        "p3_f32_nan_inf_hits": 0,
        "unordered_active_hits": 0,
        "unordered_negative_mentions": unordered_negative_mentions,
        "docs_historical_mentions": docs_historical_mentions,
    }


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

    if args.mode not in {
        "p0-baseline",
        "p1-migrated",
        "p1-general-alu-lock",
        "p2-bitcast-const-lock",
        "p3-general-cmp-lock",
    }:
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
    if args.mode in {
        "p1-general-alu-lock",
        "p2-bitcast-const-lock",
        "p3-general-cmp-lock",
    }:
        matrix_counts["special_case_removed_in_p1"] = len(SPECIAL_CASE_MATRIX)
    special_case_counts = audit_special_case_presence(repo_root, matrix_counts, args.mode)
    direct_counts = audit_direct_paths(repo_root)
    vc4tile_counts = audit_vc4tile(repo_root)
    tile_counts = audit_forbidden_tile_dsl(repo_root)
    fixture_counts = audit_fixture_purity(repo_root)
    legacy_user_counts = (
        audit_no_legacy_user_spellings(repo_root)
        if args.mode == "p1-migrated"
        else {}
    )
    p1_lock_counts = (
        audit_p1_general_alu_lock(repo_root)
        if args.mode
        in {"p1-general-alu-lock", "p2-bitcast-const-lock", "p3-general-cmp-lock"}
        else {}
    )
    p2_lock_counts = (
        audit_p2_bitcast_const_lock(repo_root, matrix)
        if args.mode == "p2-bitcast-const-lock"
        else {}
    )
    p3_lock_counts = (
        audit_p3_general_cmp_lock(repo_root, matrix)
        if args.mode == "p3-general-cmp-lock"
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
    if p1_lock_counts:
        print(f"p1_general_alu_lock: {format_counts(p1_lock_counts)}")
    if p2_lock_counts:
        print(f"p2_bitcast_const_lock: {format_counts(p2_lock_counts)}")
    if p3_lock_counts:
        print(f"p3_general_cmp_lock: {format_counts(p3_lock_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
