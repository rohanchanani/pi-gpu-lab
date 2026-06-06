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
    "removed_in_p7",
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
    "implemented_pending_hardware",
    "hardware_proven_pending_final_acceptance",
    "accepted",
    "removed_in_p4",
}

P3_POST_PHASE_STAGED_STATUSES = {
    "p6_memory_path_coherency_policy": {
        "implemented_pending_migration",
        "hardware_proven_pending_full_migration",
        "hardware_proven_pending_final_acceptance",
        "accepted",
    },
    "p7_tmu_load_explicit_safe_inactive_offset": {
        "implemented_pending_hardware",
        "hardware_proven_pending_migration",
        "hardware_proven_pending_final_acceptance",
        "accepted",
    },
    "p7_remove_old_tmu_load_signature": {
        "migration_target",
        "removed_in_p7",
    },
    "p8_vdw_store_inactive_preserve_full_tail_rect": {
        "implemented_pending_hardware",
        "hardware_proven_pending_migration",
        "hardware_proven_pending_final_acceptance",
        "accepted",
    },
    "p9_fragment_pack": {
        "implemented_pending_hardware",
        "hardware_proven_pending_final_acceptance",
    },
    "p9_fragment_unpack": {
        "implemented_pending_hardware",
        "hardware_proven_pending_final_acceptance",
    },
    "p9_vpm_subword_w16_w8_modes": {
        "implemented_pending_hardware",
        "hardware_proven_pending_final_acceptance",
    },
    "p9_vdr_vdw_subword_dma_modes": {
        "implemented_pending_hardware",
        "hardware_proven_pending_final_acceptance",
    },
}

P4_I32_REDUCE_KINDS = {
    "add",
    "min_s",
    "max_s",
    "min_u",
    "max_u",
    "bit_and",
    "bit_or",
    "bit_xor",
}

P4_F32_REDUCE_KINDS = {
    "add",
    "fmin",
    "fmax",
}

P5_ACCEPTED_ARITH_OPS = {
    "arith.constant",
    "arith.addi",
    "arith.subi",
    "arith.muli",
    "arith.shli",
    "arith.shrui",
    "arith.shrsi",
    "arith.andi",
    "arith.ori",
    "arith.xori",
    "arith.minsi",
    "arith.maxsi",
    "arith.minui",
    "arith.maxui",
    "arith.cmpi",
    "arith.select",
    "arith.bitcast",
    "arith.extui",
    "arith.trunci",
}

P5_REJECTED_ARITH_OPS = {
    "arith.addf",
    "arith.subf",
    "arith.mulf",
    "arith.divf",
    "arith.minimumf",
    "arith.maximumf",
    "arith.minnumf",
    "arith.maxnumf",
    "arith.divsi",
    "arith.divui",
    "arith.remsi",
    "arith.remui",
    "arith.sitofp",
    "arith.fptosi",
    "arith.uitofp",
    "arith.fptoui",
    "arith.index_cast",
}

P5_POST_PHASES = {
    "P6",
    "P7",
    "P8",
    "P9",
    "P10",
    "P11",
    "P12",
    "P13",
    "DEFERRED_SPARSE_VDW_STORE",
}

P8_5_MIXED_ACCEPTANCE_MANIFEST = Path(
    "compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/"
    "mixed_acceptance_manifest.json"
)
P8_5_MIXED_ACCEPTANCE_CHECKER = Path(
    "compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/"
    "check_mixed_acceptance_coverage.py"
)
P8_5_MIXED_ACCEPTANCE_RUNNER = Path(
    "compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/"
    "run_mixed_acceptance.sh"
)
P8_5_MIXED_POLICY_DOC = Path(
    "compiler/docs/vc4kernel_mixed_acceptance_policy.md"
)
P8_5_REQUIRED_MIXED_FIXTURES = {
    "mixed_elementwise_tmu_alu_cmp_vdw_vc4kernel",
    "mixed_i32_control_reduce_scalar_address_vc4kernel",
    "mixed_f32_reduce_tmu_loop_spill_vdw_vc4kernel",
    "mixed_blocked_gemv_vpm_fullstack_vc4kernel",
    "mixed_blocked_gemm_vpm_fullstack_vc4kernel",
    "mixed_vertical_rect_vpm_vdw_preserve_vc4kernel",
    "mixed_cooperative_barrier_vpm_transpose_vc4kernel",
    "mixed_lower_half_spill_dma_branch_ssavc4",
}
P8_5_REPRESENTATIVE_ISOLATED_FIXTURES = {
    "fragment_alu_add_i32_logic_shift_vc4kernel",
    "fragment_cmp_f32_finite_ordered_vc4kernel",
    "fragment_reduce_f32_add_finite_tree_vc4kernel",
    "scalar_i32_address_math_vc4kernel",
    "memory_path_explicit_tmu_vdw_vc4kernel",
    "tmu_load_safe_offset_full_tail_vc4kernel",
    "vdw_store_fragment_preserve_full_tail_vc4kernel",
    "vdw_store_rect_from_vpm_preserve_dynamic_shape_vc4kernel",
}

P5_POST_PHASE_ALLOWED_STATUSES = {
    "planned",
    "migration_target",
    "deterministic_reject",
}

P5_POST_PHASE_STAGED_STATUSES = {
    "p6_memory_path_coherency_policy": {
        "implemented_pending_migration",
        "hardware_proven_pending_full_migration",
        "hardware_proven_pending_final_acceptance",
        "accepted",
    },
    "p7_tmu_load_explicit_safe_inactive_offset": {
        "implemented_pending_hardware",
        "hardware_proven_pending_migration",
        "hardware_proven_pending_final_acceptance",
        "accepted",
    },
    "p7_remove_old_tmu_load_signature": {
        "migration_target",
        "removed_in_p7",
    },
    "p8_vdw_store_inactive_preserve_full_tail_rect": {
        "implemented_pending_hardware",
        "hardware_proven_pending_migration",
        "hardware_proven_pending_final_acceptance",
        "accepted",
    },
    "p9_fragment_pack": {
        "implemented_pending_hardware",
        "hardware_proven_pending_final_acceptance",
    },
    "p9_fragment_unpack": {
        "implemented_pending_hardware",
        "hardware_proven_pending_final_acceptance",
    },
    "p9_vpm_subword_w16_w8_modes": {
        "implemented_pending_hardware",
        "hardware_proven_pending_final_acceptance",
    },
    "p9_vdr_vdw_subword_dma_modes": {
        "implemented_pending_hardware",
        "hardware_proven_pending_final_acceptance",
    },
}

P6_MEMORY_OP_POLICIES = {
    "vc4kernel.tmu_load_fragment": (
        "#vc4kernel.memory_path<tmu_global_read>",
        "#vc4kernel.coherency<readonly_tmu>",
    ),
    "vc4kernel.vpm_write_fragment": (
        "#vc4kernel.memory_path<vpm_qpu>",
        "#vc4kernel.coherency<vpm_local>",
    ),
    "vc4kernel.vpm_read_fragment": (
        "#vc4kernel.memory_path<vpm_qpu>",
        "#vc4kernel.coherency<vpm_local>",
    ),
    "vc4kernel.vdr_load_to_vpm": (
        "#vc4kernel.memory_path<vdr_global_to_vpm>",
        "#vc4kernel.coherency<dma_ordered>",
    ),
    "vc4kernel.vdr_load_rect_to_vpm": (
        "#vc4kernel.memory_path<vdr_global_to_vpm>",
        "#vc4kernel.coherency<dma_ordered>",
    ),
    "vc4kernel.vdw_store_fragment": (
        "#vc4kernel.memory_path<vdw_global_store>",
        "#vc4kernel.coherency<dma_ordered>",
    ),
    "vc4kernel.vdw_store_vpm_fragment": (
        "#vc4kernel.memory_path<vdw_global_store>",
        "#vc4kernel.coherency<dma_ordered>",
    ),
    "vc4kernel.vdw_store_rect_from_vpm": (
        "#vc4kernel.memory_path<vdw_global_store>",
        "#vc4kernel.coherency<dma_ordered>",
    ),
}

P6_TARGETED_MEMORY_FIXTURES = {
    "memory_path_explicit_tmu_vdw_vc4kernel": {
        "vc4kernel.tmu_load_fragment": (
            "#vc4kernel.memory_path<tmu_global_read>",
            "#vc4kernel.coherency<readonly_tmu>",
        ),
        "vc4kernel.vdw_store_fragment": (
            "#vc4kernel.memory_path<vdw_global_store>",
            "#vc4kernel.coherency<dma_ordered>",
        ),
    },
    "memory_path_explicit_vdr_vpm_vdw_vc4kernel": {
        "vc4kernel.vdr_load_rect_to_vpm": (
            "#vc4kernel.memory_path<vdr_global_to_vpm>",
            "#vc4kernel.coherency<dma_ordered>",
        ),
        "vc4kernel.vpm_read_fragment": (
            "#vc4kernel.memory_path<vpm_qpu>",
            "#vc4kernel.coherency<vpm_local>",
        ),
        "vc4kernel.vpm_write_fragment": (
            "#vc4kernel.memory_path<vpm_qpu>",
            "#vc4kernel.coherency<vpm_local>",
        ),
        "vc4kernel.vdw_store_rect_from_vpm": (
            "#vc4kernel.memory_path<vdw_global_store>",
            "#vc4kernel.coherency<dma_ordered>",
        ),
    },
    "memory_path_explicit_vpm_qpu_vc4kernel": {
        "vc4kernel.vpm_read_fragment": (
            "#vc4kernel.memory_path<vpm_qpu>",
            "#vc4kernel.coherency<vpm_local>",
        ),
        "vc4kernel.vpm_write_fragment": (
            "#vc4kernel.memory_path<vpm_qpu>",
            "#vc4kernel.coherency<vpm_local>",
        ),
        "vc4kernel.vdw_store_fragment": (
            "#vc4kernel.memory_path<vdw_global_store>",
            "#vc4kernel.coherency<dma_ordered>",
        ),
    },
}

P8_VDW_STORE_OPS = {
    "vc4kernel.vdw_store_fragment": {"full", "empty", "tail"},
    "vc4kernel.vdw_store_vpm_fragment": {"full", "empty", "tail"},
    "vc4kernel.vdw_store_rect_from_vpm": {"rect"},
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
        "p4-general-reduce-lock",
        "p5-scalar-arith-lock",
        "p6-memory-policy-targeted",
        "p6-memory-policy-lock",
        "p7-tmu-safe-load-lock",
        "p8-vdw-store-policy-lock",
        "p8-5-mixed-acceptance-lock",
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
        "p4-general-reduce-lock",
        "p5-scalar-arith-lock",
        "p6-memory-policy-targeted",
        "p6-memory-policy-lock",
        "p7-tmu-safe-load-lock",
        "p8-vdw-store-policy-lock",
        "p8-5-mixed-acceptance-lock",
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
            staged_statuses = P3_POST_PHASE_STAGED_STATUSES.get(
                feature.get("id"), set()
            )
            if (status not in P3_POST_PHASE_ALLOWED_STATUSES and
                    status not in staged_statuses):
                fail(
                    f"P3 lock expected post-P3 feature {feature.get('id')} "
                    "to remain planned/migration_target/deterministic_reject "
                    "or an explicitly staged current phase status"
                )
    if mode == "p4-general-reduce-lock":
        p4_reduce = require_matrix_feature_status_in(
            features,
            "p4_fragment_reduce_general",
            "P4",
            {"accepted", "hardware_proven_pending_final_acceptance"},
        )
        p4_reduce_text = json.dumps(p4_reduce).lower()
        for token in [
            "finite_tree",
            "min_s",
            "max_s",
            "min_u",
            "max_u",
            "bit_and",
            "bit_or",
            "bit_xor",
            "fmin",
            "fmax",
            "deterministic",
        ]:
            if token not in p4_reduce_text:
                fail(f"P4 reduction matrix entry must document {token}")
        require_matrix_feature(
            features,
            "p4_remove_add_only_reduce_specialness",
            "P4",
            "removed_in_p4",
        )
    p5_status_entries = 0
    p5_post_phase_planned_entries = 0
    if mode == "p5-scalar-arith-lock":
        p5_scalar = require_matrix_feature(
            features,
            "p5_scalar_arith_bitwise_shift_minmax_casts",
            "P5",
            "accepted",
        )
        p5_status_entries = 1
        p5_text = json.dumps(p5_scalar).lower()
        for token in [
            "exact mul32",
            "bitcast",
            "i1",
            "extui",
            "trunci",
            "numeric casts",
            "deterministic",
            "vector arith",
        ]:
            if token not in p5_text:
                fail(f"P5 scalar matrix entry must document {token}")
        for feature in features.values():
            phase = feature.get("phase")
            if phase not in P5_POST_PHASES:
                continue
            p5_post_phase_planned_entries += 1
            status = feature.get("current_status")
            staged_statuses = P5_POST_PHASE_STAGED_STATUSES.get(
                feature.get("id"), set()
            )
            if (status not in P5_POST_PHASE_ALLOWED_STATUSES and
                    status not in staged_statuses):
                fail(
                    f"P5 lock expected post-P5 feature {feature.get('id')} "
                    "to remain planned/migration_target/deterministic_reject "
                    "or an explicitly staged current phase status"
                )
    if mode == "p6-memory-policy-lock":
        p6_memory = require_matrix_feature_status_in(
            features,
            "p6_memory_path_coherency_policy",
            "P6",
            {"hardware_proven_pending_final_acceptance", "accepted"},
        )
        p6_text = json.dumps(p6_memory).lower()
        for token in [
            "memory_path",
            "coherency",
            "required",
            "inactive_load",
            "inactive_store",
            "p7",
            "p8",
            "spill",
            "vdr",
            "vpm",
            "tmu",
            "vdw",
        ]:
            if token not in p6_text:
                fail(f"P6 memory policy matrix entry must document {token}")
    if mode == "p7-tmu-safe-load-lock":
        p7_tmu = require_matrix_feature_status_in(
            features,
            "p7_tmu_load_explicit_safe_inactive_offset",
            "P7",
            {"hardware_proven_pending_final_acceptance", "accepted"},
        )
        p7_text = json.dumps(p7_tmu).lower()
        for token in [
            "safe_offset",
            "inactive_load<zero>",
            "memory_path",
            "coherency",
            "straight-line",
            "zero",
            "hardware",
        ]:
            if token not in p7_text:
                fail(f"P7 TMU safe load matrix entry must document {token}")
        require_matrix_feature(
            features,
            "p7_remove_old_tmu_load_signature",
            "P7",
            "removed_in_p7",
        )
    if mode == "p8-vdw-store-policy-lock":
        p8_vdw = require_matrix_feature_status_in(
            features,
            "p8_vdw_store_inactive_preserve_full_tail_rect",
            "P8",
            {"hardware_proven_pending_final_acceptance", "accepted"},
        )
        p8_sparse = require_matrix_feature(
            features,
            "p8_sparse_vdw_store_deterministic_reject",
            "P8",
            "deterministic_reject",
        )
        p8_text = json.dumps([p8_vdw, p8_sparse]).lower()
        for token in [
            "inactive_store<preserve>",
            "required",
            "full",
            "tail",
            "rect",
            "hardware",
            "sparse",
            "removed_in_p8",
            "read-modify-write",
        ]:
            if token not in p8_text:
                fail(f"P8 VDW store matrix entries must document {token}")
    if mode == "p8-5-mixed-acceptance-lock":
        p8_5_policy = require_matrix_feature(
            features,
            "p8_5_mixed_acceptance_policy",
            "P8_5",
            "accepted",
        )
        p8_5_text = json.dumps(p8_5_policy).lower()
        for token in [
            "mixed",
            "manifest",
            "runner",
            "lock",
            "isolated fixtures",
            "triage",
            "p9-p13",
            "hardware",
        ]:
            if token not in p8_5_text:
                fail(f"P8.5 mixed acceptance matrix entry must document {token}")
    if mode != "p4-general-reduce-lock":
        require_matrix_feature_status_in(
            features,
            "p4_remove_add_only_reduce_specialness",
            "P4",
            {"migration_target", "removed_in_p4"},
        )
    if mode not in {
        "p7-tmu-safe-load-lock",
        "p8-vdw-store-policy-lock",
        "p8-5-mixed-acceptance-lock",
    }:
        require_matrix_feature_status_in(
            features,
            "p7_remove_old_tmu_load_signature",
            "P7",
            {"migration_target", "removed_in_p7"},
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
        "p5_locked_status_entries": p5_status_entries,
        "p5_post_phase_planned_entries": p5_post_phase_planned_entries,
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
        "p4-general-reduce-lock",
        "p5-scalar-arith-lock",
        "p6-memory-policy-targeted",
        "p6-memory-policy-lock",
        "p7-tmu-safe-load-lock",
        "p8-vdw-store-policy-lock",
        "p8-5-mixed-acceptance-lock",
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


def collect_fragment_reduce_ops(text):
    reduce_ops = []
    for line in text.splitlines():
        if "vc4kernel.fragment_reduce" not in line:
            continue
        if "// CHECK:" in line:
            continue
        kind_match = re.search(r"#vc4kernel\.reduce<([^>]+)>", line)
        reduce_ops.append(
            {
                "line": line,
                "kind": kind_match.group(1) if kind_match else None,
                "has_finite_tree_policy": "#vc4kernel.fp_reduce_policy<finite_tree>" in line,
                "has_f32_type": "vector<16xf32>" in line,
                "has_i32_type": "vector<16xi32>" in line,
            }
        )
    return reduce_ops


def audit_p4_general_reduce_lock(repo_root, matrix):
    features = feature_by_id(matrix)
    require_matrix_feature_status_in(
        features,
        "p4_fragment_reduce_general",
        "P4",
        {"accepted", "hardware_proven_pending_final_acceptance"},
    )
    require_matrix_feature(
        features,
        "p4_remove_add_only_reduce_specialness",
        "P4",
        "removed_in_p4",
    )

    source_roots = [
        Path("compiler/include/vc4/Dialect/VC4Kernel"),
        Path("compiler/lib/Dialect/VC4Kernel"),
        Path("compiler/lib/Conversion/VC4KernelToSSAVC4"),
    ]
    stale_source_hits = []
    stale_phrases = (
        "only add reductions are supported",
        "add-only",
        "add only",
        "only valid kind",
    )
    for root in source_roots:
        for path in iter_text_files(repo_root / root, repo_root):
            text = read_text(path).lower()
            if "fragment_reduce" not in text and "reduction" not in text:
                continue
            for phrase in stale_phrases:
                if phrase in text:
                    stale_source_hits.append((path, phrase))
    if stale_source_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{phrase}"
            for path, phrase in stale_source_hits[:20]
        )
        fail(f"P4 lock found stale add-only reduction source text: {details}")

    roots = [
        Path("compiler/include/vc4/Dialect/VC4Kernel"),
        Path("compiler/lib/Dialect/VC4Kernel"),
        Path("compiler/lib/Conversion/VC4KernelToSSAVC4"),
        Path("compiler/test/Dialect/VC4Kernel"),
        Path("compiler/test/Conversion/VC4KernelToSSAVC4"),
        Path("compiler/test/CodeGen/VC4Kernel/Hardware/Run"),
        Path("compiler/docs"),
    ]
    active_reduce_uses = 0
    i32_reduce_uses = 0
    f32_finite_tree_uses = 0
    f32_missing_policy_hits = []
    fp_policy_on_i32_hits = []
    unclassified_hits = []
    docs_historical_mentions = 0
    negative_missing_policy_mentions = 0
    files_scanned = 0
    historical_markers = ("historical", "removed", "old", "removed_in_p4")

    for root in roots:
        for path in iter_text_files(repo_root / root, repo_root):
            if is_support_audit_text(path):
                continue
            files_scanned += 1
            is_negative = is_negative_test(path)
            is_doc = "compiler/docs" in str(rel(path, repo_root))
            for reduce_op in collect_fragment_reduce_ops(read_text(path)):
                kind = reduce_op["kind"]
                line_lower = reduce_op["line"].lower()
                if is_doc and any(marker in line_lower for marker in historical_markers):
                    docs_historical_mentions += 1
                    continue
                if reduce_op["has_f32_type"]:
                    if kind not in P4_F32_REDUCE_KINDS:
                        if is_negative:
                            continue
                        unclassified_hits.append((path, kind))
                        continue
                    if not reduce_op["has_finite_tree_policy"]:
                        if is_negative:
                            negative_missing_policy_mentions += 1
                            continue
                        f32_missing_policy_hits.append((path, kind))
                        continue
                    active_reduce_uses += 1
                    f32_finite_tree_uses += 1
                    continue
                if reduce_op["has_i32_type"]:
                    if kind not in P4_I32_REDUCE_KINDS:
                        if is_negative:
                            continue
                        unclassified_hits.append((path, kind))
                        continue
                    if reduce_op["has_finite_tree_policy"]:
                        if is_negative:
                            continue
                        fp_policy_on_i32_hits.append((path, kind))
                        continue
                    active_reduce_uses += 1
                    i32_reduce_uses += 1
                    continue
                if kind is None and (is_negative or is_doc):
                    continue
                if kind is None and path.suffix not in {".mlir", ".md"}:
                    continue
                if is_doc and not (reduce_op["has_f32_type"] or reduce_op["has_i32_type"]):
                    continue
                if is_negative:
                    continue
                unclassified_hits.append((path, kind))

    if f32_missing_policy_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{kind}"
            for path, kind in f32_missing_policy_hits[:20]
        )
        fail(f"P4 lock found f32 fragment_reduce without finite_tree policy: {details}")
    if fp_policy_on_i32_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{kind}"
            for path, kind in fp_policy_on_i32_hits[:20]
        )
        fail(f"P4 lock found fp_reduce_policy on i32 fragment_reduce: {details}")
    if unclassified_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{kind}"
            for path, kind in unclassified_hits[:20]
        )
        fail(f"P4 lock found unclassified fragment_reduce use: {details}")
    if i32_reduce_uses == 0 or f32_finite_tree_uses == 0:
        fail("P4 lock expected both i32 and finite_tree f32 fragment_reduce uses")

    fixture_root = repo_root / "compiler/test/CodeGen/VC4Kernel/Hardware/Run"
    nan_inf_re = re.compile(r"\b(?:nan|inf|infinity|NAN|INF|INFINITY)\b")
    p4_f32_nan_inf_hits = []
    for path in sorted(fixture_root.rglob("input.mlir")):
        text = read_text(path)
        if "vc4kernel.fragment_reduce" not in text or "vector<16xf32>" not in text:
            continue
        if nan_inf_re.search(text):
            p4_f32_nan_inf_hits.append(path)
    if p4_f32_nan_inf_hits:
        details = "; ".join(
            str(rel(path, repo_root)) for path in p4_f32_nan_inf_hits[:20]
        )
        fail(f"P4 lock found NaN/Inf text in f32 reduction fixture input: {details}")

    return {
        "active_reduce_uses": active_reduce_uses,
        "docs_historical_mentions": docs_historical_mentions,
        "f32_finite_tree_reduce_uses": f32_finite_tree_uses,
        "f32_missing_policy_hits": 0,
        "fp_policy_on_i32_hits": 0,
        "i32_reduce_uses": i32_reduce_uses,
        "negative_missing_policy_mentions": negative_missing_policy_mentions,
        "p4_f32_nan_inf_hits": 0,
        "reduce_files_scanned": files_scanned,
        "stale_add_only_source_hits": 0,
    }


def audit_p5_scalar_arith_lock(repo_root, matrix):
    features = feature_by_id(matrix)
    require_matrix_feature(
        features,
        "p5_scalar_arith_bitwise_shift_minmax_casts",
        "P5",
        "accepted",
    )

    lowering_path = (
        repo_root
        / "compiler/lib/Conversion/VC4KernelToSSAVC4/VC4KernelToSSAVC4.cpp"
    )
    lowering_text = read_text(lowering_path)
    required_lowering_tokens = [
        "emitI32Mul32Fallback",
        "Exact modulo-2^32 multiply",
        "builder.getI32IntegerAttr(0xffff)",
        "builder.getI32IntegerAttr(16)",
        "arith.sitofp requires exact scalar numeric cast support not available in P5",
        "arith.fptosi requires exact scalar numeric cast support not available in P5",
        "scalar numeric casts require exact scalar numeric cast support not available in P5",
        "scalar f32 arithmetic is not supported in vc4kernel scalar arith",
        "integer division and remainder are not supported in vc4kernel scalar arith",
        "arith.bitcast requires scalar i32/f32 reinterpretation in vc4kernel",
        "arith.extui in vc4kernel supports only i1 to i32 in P5",
        "arith.trunci in vc4kernel supports only i32 to i1 low-bit trunc in P5",
    ]
    for token in required_lowering_tokens:
        if token not in lowering_text:
            fail(f"P5 scalar arith lock missing lowering/verifier token: {token}")
    muli_match = re.search(
        r'if \(hasName\(op, "arith\.muli"\)\) \{(?P<body>.*?)return success\(\);',
        lowering_text,
        re.DOTALL,
    )
    if not muli_match or "emitI32Mul32Fallback" not in muli_match.group("body"):
        fail("P5 scalar arith lock requires arith.muli to use exact fallback")
    bitcast_match = re.search(
        r'if \(hasName\(op, "arith\.bitcast"\)\) \{(?P<body>.*?)return success\(\);',
        lowering_text,
        re.DOTALL,
    )
    if not bitcast_match or "kSSAVC4MovOpName" not in bitcast_match.group("body"):
        fail("P5 scalar arith lock requires arith.bitcast to lower through mov")
    if "arith.bitcast" in lowering_text and (
        "arith.sitofp" in bitcast_match.group("body")
        or "arith.fptosi" in bitcast_match.group("body")
    ):
        fail("P5 scalar arith lock found numeric conversion in bitcast lowering")

    roots = [
        Path("compiler/include/vc4/Dialect/VC4Kernel"),
        Path("compiler/lib/Dialect/VC4Kernel"),
        Path("compiler/lib/Conversion/VC4KernelToSSAVC4"),
        Path("compiler/test/Dialect/VC4Kernel"),
        Path("compiler/test/Conversion/VC4KernelToSSAVC4"),
        Path("compiler/test/CodeGen/VC4Kernel/Hardware/Run"),
    ]
    active_arith_ops = Counter()
    rejected_negative_mentions = Counter()
    vector_arith_hits = []
    unsupported_active_hits = []
    unsupported_type_hits = []
    files_scanned = 0
    arith_op_re = re.compile(r"\barith\.[A-Za-z0-9_]+")
    unsupported_type_re = re.compile(
        r"\barith\.[A-Za-z0-9_]+[^\n]*(?:\bi8\b|\bi16\b|\bi64\b|\bf64\b|\bindex\b)"
    )
    vector_arith_re = re.compile(r"\barith\.[A-Za-z0-9_]+[^\n]*vector<")
    for root in roots:
        for path in iter_text_files(repo_root / root, repo_root):
            if is_support_audit_text(path):
                continue
            files_scanned += 1
            if path.suffix != ".mlir":
                continue
            text = read_text(path)
            negative = is_negative_test(path)
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("//") or stripped.startswith("#"):
                    continue
                ops = arith_op_re.findall(line)
                if not ops:
                    continue
                if negative:
                    for op_name in ops:
                        if op_name in P5_REJECTED_ARITH_OPS:
                            rejected_negative_mentions[op_name] += 1
                    continue
                for op_name in ops:
                    if op_name not in P5_ACCEPTED_ARITH_OPS:
                        unsupported_active_hits.append((path, op_name, stripped))
                    else:
                        active_arith_ops[op_name] += 1
                if vector_arith_re.search(line):
                    vector_arith_hits.append((path, stripped))
                if unsupported_type_re.search(line):
                    unsupported_type_hits.append((path, stripped))

    if unsupported_active_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{op_name}:{line}"
            for path, op_name, line in unsupported_active_hits[:20]
        )
        fail(f"P5 scalar arith lock found unsupported active arith op: {details}")
    if vector_arith_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{line}" for path, line in vector_arith_hits[:20]
        )
        fail(f"P5 scalar arith lock found active vector arith: {details}")
    if unsupported_type_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{line}" for path, line in unsupported_type_hits[:20]
        )
        fail(f"P5 scalar arith lock found unsupported scalar width/type: {details}")

    required_active_ops = {
        "arith.muli",
        "arith.andi",
        "arith.ori",
        "arith.xori",
        "arith.shrui",
        "arith.shrsi",
        "arith.minsi",
        "arith.maxsi",
        "arith.minui",
        "arith.maxui",
        "arith.bitcast",
        "arith.extui",
        "arith.trunci",
    }
    missing_active = sorted(op for op in required_active_ops if active_arith_ops[op] == 0)
    if missing_active:
        fail(
            "P5 scalar arith lock expected active tests/fixtures for: "
            + ", ".join(missing_active)
        )

    required_negative_ops = {
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divsi",
        "arith.divui",
        "arith.remsi",
        "arith.remui",
        "arith.sitofp",
        "arith.fptosi",
        "arith.uitofp",
        "arith.fptoui",
    }
    missing_negative = sorted(
        op for op in required_negative_ops if rejected_negative_mentions[op] == 0
    )
    if missing_negative:
        fail(
            "P5 scalar arith lock expected deterministic-reject tests for: "
            + ", ".join(missing_negative)
        )

    fixture_root = repo_root / "compiler/test/CodeGen/VC4Kernel/Hardware/Run"
    fixture_scalar_ops = Counter()
    for path in sorted(fixture_root.rglob("input.mlir")):
        text = read_text(path)
        if "vc4kernel.kernel" not in text:
            continue
        for op_name in arith_op_re.findall(text):
            fixture_scalar_ops[op_name] += 1
    for op_name in ["arith.muli", "arith.bitcast", "arith.extui", "arith.trunci"]:
        if fixture_scalar_ops[op_name] == 0:
            fail(f"P5 scalar arith lock expected hardware fixture use of {op_name}")

    return {
        "active_arith_ops": sum(active_arith_ops.values()),
        "deterministic_reject_tests": sum(rejected_negative_mentions.values()),
        "exact_muli_fallback": 1,
        "files_scanned": files_scanned,
        "fixture_scalar_ops": sum(fixture_scalar_ops.values()),
        "numeric_cast_rejects": rejected_negative_mentions["arith.sitofp"]
        + rejected_negative_mentions["arith.fptosi"]
        + rejected_negative_mentions["arith.uitofp"]
        + rejected_negative_mentions["arith.fptoui"],
        "unsupported_active_hits": 0,
        "unsupported_type_hits": 0,
        "vector_arith_hits": 0,
    }


def audit_p6_memory_policy_targeted(repo_root, matrix):
    features = feature_by_id(matrix)
    p6_feature = require_matrix_feature_status_in(
        features,
        "p6_memory_path_coherency_policy",
        "P6",
        {
            "implemented_pending_migration",
            "hardware_proven_pending_full_migration",
            "hardware_proven_pending_final_acceptance",
            "accepted",
        },
    )
    p6_text = json.dumps(p6_feature).lower()
    for token in [
        "memory_path",
        "coherency",
        "spill",
        "vdr",
        "vpm",
        "tmu",
        "vdw",
        "p7",
        "p8",
    ]:
        if token not in p6_text:
            fail(f"P6 memory policy matrix entry must document {token}")

    ssavc4_to_vc4 = read_text(
        repo_root / "compiler/lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp"
    )
    reload_start = ssavc4_to_vc4.find("void RawVDRSpillReloadSequence::emit")
    reload_end = ssavc4_to_vc4.find("void SpillActionSequence::emit", reload_start)
    if reload_start < 0 or reload_end < 0:
        fail("P6 memory policy audit expected RawVDRSpillReloadSequence emission")
    reload_block = ssavc4_to_vc4[reload_start:reload_end].lower()
    if "emitrawvdrload" not in reload_block or "createvpmvcdsetup" not in reload_block:
        fail("P6 memory policy audit expected spill reload through VDR and VPM")
    if re.search(r"\b(?:tmu|tmu0|ldtmu0)\b", reload_block):
        fail("P6 memory policy audit found TMU/ldtmu0 in hidden spill reload path")

    fixture_root = repo_root / "compiler/test/CodeGen/VC4Kernel/Hardware/Run"
    missing_fixtures = []
    attr_violations = []
    op_counts = Counter()
    for fixture, expected_ops in P6_TARGETED_MEMORY_FIXTURES.items():
        path = fixture_root / fixture / "input.mlir"
        if not path.is_file():
            missing_fixtures.append(fixture)
            continue
        text = read_text(path)
        if "vc4kernel.kernel" not in text:
            attr_violations.append(f"{rel(path, repo_root)}: missing vc4kernel.kernel")
        if "ssavc4." in text or "vc4.qpu." in text or "vc4.module" in text:
            attr_violations.append(
                f"{rel(path, repo_root)}: contains source-authored lower-half ops"
            )
        if "compiler_spill_vdw_vdr" in text or "compiler_spill_coherent" in text:
            attr_violations.append(
                f"{rel(path, repo_root)}: user fixture contains internal compiler spill attr"
            )
        fixture_counts = Counter()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            for op_name, (memory_path, coherency) in expected_ops.items():
                if op_name not in stripped:
                    continue
                fixture_counts[op_name] += 1
                op_counts[op_name] += 1
                if memory_path not in stripped:
                    attr_violations.append(
                        f"{rel(path, repo_root)}: {op_name} missing {memory_path}"
                    )
                if coherency not in stripped:
                    attr_violations.append(
                        f"{rel(path, repo_root)}: {op_name} missing {coherency}"
                    )
                if "memory_path" not in stripped or "coherency" not in stripped:
                    attr_violations.append(
                        f"{rel(path, repo_root)}: {op_name} missing explicit attrs"
                    )
        for op_name in expected_ops:
            if fixture_counts[op_name] == 0:
                attr_violations.append(
                    f"{rel(path, repo_root)}: expected memory op {op_name}"
                )
    if missing_fixtures:
        fail("P6 memory policy audit missing targeted fixtures: " + ", ".join(missing_fixtures))
    if attr_violations:
        fail("P6 targeted memory fixture violations: " + "; ".join(attr_violations[:20]))

    internal_attr_hits = []
    for path in sorted(fixture_root.rglob("input.mlir")):
        text = read_text(path)
        if "compiler_spill_vdw_vdr" in text or "compiler_spill_coherent" in text:
            internal_attr_hits.append(path)
    if internal_attr_hits:
        details = "; ".join(str(rel(path, repo_root)) for path in internal_attr_hits[:20])
        fail(f"P6 memory policy audit found internal spill attrs on user fixtures: {details}")

    blocked_tmu_hits = []
    for fixture in ["gemm_blocked_vpm_vc4kernel", "gemv_blocked_vpm_vc4kernel"]:
        path = fixture_root / fixture / "input.mlir"
        if not path.is_file():
            continue
        for line in read_text(path).splitlines():
            if "vc4kernel.tmu_load_fragment %a" in line or (
                fixture.startswith("gemm_")
                and "vc4kernel.tmu_load_fragment %b" in line
            ):
                blocked_tmu_hits.append(path)
                break
    if blocked_tmu_hits:
        details = "; ".join(str(rel(path, repo_root)) for path in blocked_tmu_hits)
        fail(f"P6 memory policy audit found TMU in blocked VPM GEMM/GEMV fixtures: {details}")

    return {
        "blocked_tmu_fixture_hits": 0,
        "hidden_spill_reload_tmu_hits": 0,
        "internal_spill_user_attr_hits": 0,
        "p6_targeted_fixture_count": len(P6_TARGETED_MEMORY_FIXTURES),
        "p6_targeted_memory_ops": sum(op_counts.values()),
    }


def audit_p6_memory_policy_lock(repo_root, matrix):
    counts = audit_p6_memory_policy_targeted(repo_root, matrix)

    roots = [
        Path("compiler/test/Dialect/VC4Kernel"),
        Path("compiler/test/Conversion/VC4KernelToSSAVC4"),
        Path("compiler/test/CodeGen/VC4Kernel/Hardware/Run"),
    ]
    files_scanned = 0
    active_memory_ops = Counter()
    negative_memory_ops = Counter()
    missing_attr_hits = []
    mismatch_hits = []
    internal_attr_hits = []
    source_lower_half_hits = []
    producer_hits = []
    producer_prefixes = [
        "vector.",
        "memref.",
        "scf.",
        "linalg.",
        "gpu.",
        "tt.",
        "triton.",
    ]
    for root in roots:
        for path in iter_text_files(repo_root / root, repo_root):
            if path.suffix != ".mlir":
                continue
            files_scanned += 1
            negative = is_negative_test(path)
            text = read_text(path)
            if (
                "compiler_spill_vdw_vdr" in text
                or "compiler_spill_coherent" in text
            ) and not negative:
                internal_attr_hits.append(path)
            if "compiler/test/CodeGen/VC4Kernel/Hardware/Run" in str(
                rel(path, repo_root)
            ):
                if "ssavc4." in text or "vc4.qpu." in text or "vc4.module" in text:
                    source_lower_half_hits.append(path)
                for prefix in producer_prefixes:
                    if prefix in text:
                        producer_hits.append((path, prefix))
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue
                for op_name, (memory_path, coherency) in P6_MEMORY_OP_POLICIES.items():
                    if op_name not in stripped:
                        continue
                    if negative:
                        negative_memory_ops[op_name] += 1
                        continue
                    active_memory_ops[op_name] += 1
                    has_memory_path = "memory_path" in stripped
                    has_coherency = "coherency" in stripped
                    if not has_memory_path or not has_coherency:
                        missing_attr_hits.append((path, op_name, stripped))
                        continue
                    if memory_path not in stripped or coherency not in stripped:
                        mismatch_hits.append((path, op_name, stripped))
    if missing_attr_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{op_name}:{line}"
            for path, op_name, line in missing_attr_hits[:20]
        )
        fail(f"P6 memory policy lock found memory op without explicit attrs: {details}")
    if mismatch_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{op_name}:{line}"
            for path, op_name, line in mismatch_hits[:20]
        )
        fail(f"P6 memory policy lock found mismatched memory attrs: {details}")
    if internal_attr_hits:
        details = "; ".join(str(rel(path, repo_root)) for path in internal_attr_hits[:20])
        fail(f"P6 memory policy lock found internal spill attrs on user ops: {details}")
    if source_lower_half_hits:
        details = "; ".join(str(rel(path, repo_root)) for path in source_lower_half_hits[:20])
        fail(f"P6 memory policy lock found source-authored lower-half ops in fixtures: {details}")
    if producer_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{prefix}" for path, prefix in producer_hits[:20]
        )
        fail(f"P6 memory policy lock found producer dialect in fixtures: {details}")
    missing_active_families = sorted(
        op_name for op_name in P6_MEMORY_OP_POLICIES
        if active_memory_ops[op_name] == 0
    )
    if missing_active_families:
        fail(
            "P6 memory policy lock expected active migrated uses for: "
            + ", ".join(missing_active_families)
        )
    if sum(negative_memory_ops.values()) == 0:
        fail("P6 memory policy lock expected deterministic-reject memory policy tests")

    features = feature_by_id(matrix)
    require_matrix_feature_status_in(
        features,
        "p6_memory_path_coherency_policy",
        "P6",
        {"hardware_proven_pending_final_acceptance", "accepted"},
    )
    require_matrix_feature_status_in(
        features,
        "p7_remove_old_tmu_load_signature",
        "P7",
        {"migration_target", "removed_in_p7"},
    )
    require_matrix_feature(features, "p8_sparse_vdw_store_deterministic_reject",
                           "P8", "deterministic_reject")
    counts.update(
        {
            "active_memory_ops": sum(active_memory_ops.values()),
            "active_memory_op_families": len(active_memory_ops),
            "files_scanned": files_scanned,
            "internal_spill_user_attr_hits": 0,
            "memory_attr_mismatch_hits": 0,
            "memory_attr_missing_hits": 0,
            "negative_memory_policy_ops": sum(negative_memory_ops.values()),
            "producer_fixture_hits": 0,
            "source_lower_half_fixture_hits": 0,
        }
    )
    return counts


def audit_p7_tmu_safe_load_lock(repo_root, matrix):
    counts = audit_p6_memory_policy_targeted(repo_root, matrix)
    features = feature_by_id(matrix)
    require_matrix_feature_status_in(
        features,
        "p7_tmu_load_explicit_safe_inactive_offset",
        "P7",
        {"hardware_proven_pending_final_acceptance", "accepted"},
    )
    require_matrix_feature(
        features,
        "p7_remove_old_tmu_load_signature",
        "P7",
        "removed_in_p7",
    )

    source_roots = [
        Path("compiler/include/vc4/Dialect/VC4Kernel"),
        Path("compiler/lib/Dialect/VC4Kernel"),
        Path("compiler/lib/Conversion/VC4KernelToSSAVC4"),
    ]
    tmu_source_policy_hits = []
    for root in source_roots:
        for path in iter_text_files(repo_root / root, repo_root):
            text = read_text(path)
            for line_no, line in enumerate(text.splitlines(), start=1):
                lowered = line.lower()
                if "tmu" not in lowered:
                    continue
                if not re.search(r"\b(?:infer|implicit|legacy|migration-only)\b", lowered):
                    continue
                tmu_source_policy_hits.append((path, line_no, line.strip()))
    if tmu_source_policy_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{line_no}:{line}"
            for path, line_no, line in tmu_source_policy_hits[:20]
        )
        fail(f"P7 lock found old TMU safe-address policy text in active source: {details}")

    conversion = read_text(
        repo_root / "compiler/lib/Conversion/VC4KernelToSSAVC4/VC4KernelToSSAVC4.cpp"
    )
    tmu_start = conversion.rfind("if (hasName(op, kTMULoadOpName))")
    tmu_end = conversion.find("if (hasName(op, kVDWStoreOpName))", tmu_start)
    if tmu_start < 0 or tmu_end < 0:
        fail("P7 lock expected explicit TMU lowering block in VC4KernelToSSAVC4")
    tmu_block = conversion[tmu_start:tmu_end]
    for token in [
        "Value safeScalarOffset = mapValue(op, op->getOperand(3), state)",
        "kSSAVC4SplatOpName",
        "emitPredicateSelect(op, builder, *predicate, offsets, safeOffsetVec)",
        "emitTMULoadFragment(op, builder, base, *safeOffsets",
        "emitPredicateSelect(op, builder, *predicate, loaded, zero)",
    ]:
        if token not in tmu_block:
            fail(f"P7 lock expected straight-line explicit safe-offset TMU lowering token: {token}")
    for forbidden in ["createBranch", "createCondBranch", "Block *", "push_back"]:
        if forbidden in tmu_block:
            fail(
                "P7 lock found branchy CFG construction in TMU safe-offset lowering: "
                + forbidden
            )

    scan_roots = [
        Path("compiler/test/Dialect/VC4Kernel"),
        Path("compiler/test/Conversion/VC4KernelToSSAVC4"),
        Path("compiler/test/CodeGen/VC4Kernel/Hardware/Run"),
    ]
    files_scanned = 0
    active_tmu_loads = 0
    negative_tmu_loads = 0
    explicit_safe_offset_hits = 0
    inactive_zero_hits = 0
    missing_policy_hits = []
    old_signature_hits = []
    for root in scan_roots:
        for path in iter_text_files(repo_root / root, repo_root):
            if path.suffix not in {".mlir", ".test"}:
                continue
            files_scanned += 1
            negative = is_negative_test(path)
            for line_no, line in enumerate(read_text(path).splitlines(), start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue
                if "vc4kernel.tmu_load_fragment" not in stripped:
                    continue
                if negative:
                    negative_tmu_loads += 1
                    continue
                active_tmu_loads += 1
                has_safe_offset = ", i32 ->" in stripped
                has_inactive_zero = (
                    "inactive_load = #vc4kernel.inactive_load<zero>" in stripped
                )
                has_memory_path = (
                    "#vc4kernel.memory_path<tmu_global_read>" in stripped
                )
                has_coherency = "#vc4kernel.coherency<readonly_tmu>" in stripped
                if has_safe_offset:
                    explicit_safe_offset_hits += 1
                else:
                    old_signature_hits.append((path, line_no, stripped))
                if has_inactive_zero:
                    inactive_zero_hits += 1
                if not (has_safe_offset and has_inactive_zero and
                        has_memory_path and has_coherency):
                    missing_policy_hits.append((path, line_no, stripped))
    if old_signature_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{line_no}:{line}"
            for path, line_no, line in old_signature_hits[:20]
        )
        fail(f"P7 lock found active old TMU load signature: {details}")
    if missing_policy_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{line_no}:{line}"
            for path, line_no, line in missing_policy_hits[:20]
        )
        fail(f"P7 lock found active TMU load missing safe offset/inactive/P6 attrs: {details}")
    if active_tmu_loads == 0:
        fail("P7 lock expected active explicit tmu_load_fragment uses")
    if negative_tmu_loads == 0:
        fail("P7 lock expected deterministic-reject TMU load tests")

    docs = read_text(
        repo_root / "compiler/docs/codegen/vc4kernel_dialect_strict_specification.md"
    )
    for token in [
        "safe_offset",
        "inactive_load = #vc4kernel.inactive_load<zero>",
        "request base + safe_offset",
        "removed_in_p7",
    ]:
        if token not in docs:
            fail(f"P7 lock expected strict spec to document {token}")

    counts.update(
        {
            "active_tmu_loads": active_tmu_loads,
            "explicit_safe_offset_hits": explicit_safe_offset_hits,
            "files_scanned": files_scanned,
            "inactive_zero_hits": inactive_zero_hits,
            "negative_tmu_loads": negative_tmu_loads,
            "old_signature_hits": 0,
            "straight_line_tmu_lowering": 1,
            "tmu_source_policy_hits": 0,
        }
    )
    return counts


def classify_vc4kernel_predicates(text):
    kinds = {}
    ssa = r"%[A-Za-z0-9_.$-]+"
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        full = re.match(rf"({ssa})\s*=\s*vc4kernel\.pred\.full\b", stripped)
        if full:
            kinds[full.group(1)] = "full"
            continue
        empty = re.match(rf"({ssa})\s*=\s*vc4kernel\.pred\.empty\b", stripped)
        if empty:
            kinds[empty.group(1)] = "empty"
            continue
        tail = re.match(rf"({ssa})\s*=\s*vc4kernel\.pred\.tail\b", stripped)
        if tail:
            kinds[tail.group(1)] = "tail"
            continue
        rect = re.match(rf"({ssa})\s*=\s*vc4kernel\.pred\.rect\b", stripped)
        if rect:
            kinds[rect.group(1)] = "rect"
            continue
        cmp_match = re.match(rf"({ssa})\s*=\s*vc4kernel\.fragment_cmp\b", stripped)
        if cmp_match:
            kinds[cmp_match.group(1)] = "general"
            continue
        unary = re.match(
            rf"({ssa})\s*=\s*vc4kernel\.pred\.not\s+({ssa})\b", stripped
        )
        if unary:
            operand = kinds.get(unary.group(2), "unknown")
            if operand == "full":
                result = "empty"
            elif operand == "empty":
                result = "full"
            else:
                result = "general"
            kinds[unary.group(1)] = result
            continue
        binary = re.match(
            rf"({ssa})\s*=\s*vc4kernel\.pred\.(and|or)\s+({ssa}),\s*({ssa})\b",
            stripped,
        )
        if binary:
            lhs = kinds.get(binary.group(3), "unknown")
            rhs = kinds.get(binary.group(4), "unknown")
            if binary.group(2) == "and":
                if lhs == "empty" or rhs == "empty":
                    result = "empty"
                elif lhs == "full":
                    result = rhs
                elif rhs == "full":
                    result = lhs
                elif lhs == rhs:
                    result = lhs
                elif lhs == "tail" and rhs == "tail":
                    result = "tail"
                else:
                    result = "general"
            else:
                if lhs == "full" or rhs == "full":
                    result = "full"
                elif lhs == "empty":
                    result = rhs
                elif rhs == "empty":
                    result = lhs
                elif lhs == rhs:
                    result = lhs
                else:
                    result = "general"
            kinds[binary.group(1)] = result
    return kinds


def extract_vdw_store_predicate(line, op_name):
    before_attrs = line.split("{", 1)[0]
    if op_name not in before_attrs:
        return None
    args = before_attrs.split(op_name, 1)[1].strip()
    operands = [part.strip() for part in args.split(",")]
    if op_name == "vc4kernel.vdw_store_fragment":
        return operands[3] if len(operands) >= 4 else None
    if op_name == "vc4kernel.vdw_store_vpm_fragment":
        return operands[4] if len(operands) >= 5 else None
    return None


def audit_p8_vdw_store_policy_lock(repo_root, matrix):
    counts = audit_p7_tmu_safe_load_lock(repo_root, matrix)
    features = feature_by_id(matrix)
    require_matrix_feature_status_in(
        features,
        "p8_vdw_store_inactive_preserve_full_tail_rect",
        "P8",
        {"hardware_proven_pending_final_acceptance", "accepted"},
    )
    require_matrix_feature(
        features,
        "p8_sparse_vdw_store_deterministic_reject",
        "P8",
        "deterministic_reject",
    )

    scan_roots = [
        Path("compiler/test/Dialect/VC4Kernel"),
        Path("compiler/test/Conversion/VC4KernelToSSAVC4"),
        Path("compiler/test/CodeGen/VC4Kernel/Hardware/Run"),
    ]
    files_scanned = 0
    active_vdw_stores = Counter()
    negative_vdw_stores = Counter()
    missing_policy_hits = []
    sparse_hits = []
    for root in scan_roots:
        for path in iter_text_files(repo_root / root, repo_root):
            if path.suffix not in {".mlir", ".test"}:
                continue
            files_scanned += 1
            text = read_text(path)
            predicates = classify_vc4kernel_predicates(text)
            negative = is_negative_test(path)
            for line_no, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue
                for op_name, allowed_kinds in P8_VDW_STORE_OPS.items():
                    if op_name not in stripped:
                        continue
                    if negative:
                        negative_vdw_stores[op_name] += 1
                        continue
                    active_vdw_stores[op_name] += 1
                    has_inactive = (
                        "inactive_store = #vc4kernel.inactive_store<preserve>"
                        in stripped
                    )
                    has_memory_path = (
                        "#vc4kernel.memory_path<vdw_global_store>" in stripped
                    )
                    has_coherency = "#vc4kernel.coherency<dma_ordered>" in stripped
                    if not (has_inactive and has_memory_path and has_coherency):
                        missing_policy_hits.append((path, line_no, stripped))
                    if op_name == "vc4kernel.vdw_store_rect_from_vpm":
                        continue
                    pred = extract_vdw_store_predicate(stripped, op_name)
                    pred_kind = predicates.get(pred, "unknown")
                    if pred_kind not in allowed_kinds:
                        sparse_hits.append((path, line_no, op_name, pred, pred_kind))
    if missing_policy_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{line_no}:{line}"
            for path, line_no, line in missing_policy_hits[:20]
        )
        fail(f"P8 lock found VDW store missing inactive_store/P6 attrs: {details}")
    if sparse_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{line_no}:{op}:{pred}:{kind}"
            for path, line_no, op, pred, kind in sparse_hits[:20]
        )
        fail(f"P8 lock found active sparse/general VDW store: {details}")

    missing_active = sorted(
        op_name for op_name in P8_VDW_STORE_OPS if active_vdw_stores[op_name] == 0
    )
    if missing_active:
        fail("P8 lock expected active VDW store uses for: " + ", ".join(missing_active))
    if sum(negative_vdw_stores.values()) == 0:
        fail("P8 lock expected deterministic-reject VDW store tests")

    conversion = read_text(
        repo_root / "compiler/lib/Conversion/VC4KernelToSSAVC4/VC4KernelToSSAVC4.cpp"
    )
    vdw_start = conversion.rfind("if (hasName(op, kVDWStoreOpName))")
    vdw_end = conversion.find("if (hasName(op, kVPMAllocOpName))", vdw_start)
    if vdw_start < 0 or vdw_end < 0:
        fail("P8 lock expected register-fragment VDW lowering block")
    vdw_block = conversion[vdw_start:vdw_end]
    for token in [
        "VDW stores require explicit inactive_store<preserve> in Surface v2",
        "sparse VDW store masks are not supported in P8",
        "VDW preserve store requires dense contiguous/rectangular address mapping",
    ]:
        if token not in conversion:
            fail(f"P8 lock expected VDW policy diagnostic token: {token}")
    for forbidden in [
        "emitTMULoadFragment",
        "emitPredicateSelect",
        "oldValue",
        "read-modify",
        "RMW",
        "hasExplicitInactiveStorePolicy",
        "attr-absent",
        "migration-only",
    ]:
        if forbidden in vdw_block:
            fail(f"P8 lock found sparse/RMW fallback token in VDW lowering: {forbidden}")

    source_roots = [
        Path("compiler/include/vc4/Dialect/VC4Kernel"),
        Path("compiler/lib/Dialect/VC4Kernel"),
        Path("compiler/lib/Conversion/VC4KernelToSSAVC4"),
    ]
    source_policy_hits = []
    for root in source_roots:
        for path in iter_text_files(repo_root / root, repo_root):
            text = read_text(path)
            for line_no, line in enumerate(text.splitlines(), start=1):
                lowered = line.lower()
                if "vdw" not in lowered and "inactive_store" not in lowered:
                    continue
                if not re.search(
                    r"\b(?:implicit|attr-absent|migration-only|read-modify|rmw)\b",
                    lowered,
                ):
                    continue
                source_policy_hits.append((path, line_no, line.strip()))
    if source_policy_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{line_no}:{line}"
            for path, line_no, line in source_policy_hits[:20]
        )
        fail(f"P8 lock found old VDW inactive-store policy text in source: {details}")

    docs = read_text(
        repo_root / "compiler/docs/codegen/vc4kernel_dialect_strict_specification.md"
    )
    for token in [
        "inactive_store = #vc4kernel.inactive_store<preserve>",
        "VDW stores require explicit inactive_store<preserve>",
        "full, empty, and tail/active-prefix",
        "VPM-backed rectangular stores",
        "sparse VDW store masks are not supported in P8",
        "P9",
    ]:
        if token not in docs:
            fail(f"P8 lock expected strict spec to document {token}")

    counts.update(
        {
            "active_vdw_store_families": len(active_vdw_stores),
            "active_vdw_stores": sum(active_vdw_stores.values()),
            "files_scanned": files_scanned,
            "missing_inactive_store_hits": 0,
            "negative_vdw_store_tests": sum(negative_vdw_stores.values()),
            "sparse_vdw_active_hits": 0,
            "vdw_sparse_rmw_fallback_hits": 0,
            "vdw_source_policy_hits": 0,
        }
    )
    return counts


def run_mixed_acceptance_checker_lock(repo_root):
    manifest = repo_root / P8_5_MIXED_ACCEPTANCE_MANIFEST
    checker = repo_root / P8_5_MIXED_ACCEPTANCE_CHECKER
    command = [
        sys.executable,
        str(checker),
        "--manifest",
        str(manifest),
        "--repo-root",
        str(repo_root),
        "--mode",
        "lock",
    ]
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
        fail(f"mixed acceptance checker lock failed: {output}")
    return result.stdout.strip().splitlines()


def audit_p8_5_mixed_acceptance_lock(repo_root, matrix):
    counts = audit_p8_vdw_store_policy_lock(repo_root, matrix)
    features = feature_by_id(matrix)
    require_matrix_feature(
        features,
        "p8_5_mixed_acceptance_policy",
        "P8_5",
        "accepted",
    )

    required_paths = [
        P8_5_MIXED_ACCEPTANCE_MANIFEST,
        P8_5_MIXED_ACCEPTANCE_CHECKER,
        P8_5_MIXED_ACCEPTANCE_RUNNER,
        P8_5_MIXED_POLICY_DOC,
    ]
    missing_paths = [
        str(path) for path in required_paths if not (repo_root / path).is_file()
    ]
    if missing_paths:
        fail("P8.5 lock missing mixed acceptance files: " + ", ".join(missing_paths))
    if not (repo_root / P8_5_MIXED_ACCEPTANCE_RUNNER).stat().st_mode & 0o111:
        fail("P8.5 lock expected mixed acceptance runner to be executable")

    checker_summary = run_mixed_acceptance_checker_lock(repo_root)
    if not checker_summary or "mode=lock" not in checker_summary[0]:
        fail("P8.5 lock expected mixed acceptance checker lock PASS summary")

    manifest = json.loads((repo_root / P8_5_MIXED_ACCEPTANCE_MANIFEST).read_text())
    fixtures = {fixture["name"]: fixture for fixture in manifest.get("fixtures", [])}
    missing_fixtures = sorted(P8_5_REQUIRED_MIXED_FIXTURES - fixtures.keys())
    if missing_fixtures:
        fail("P8.5 lock missing mixed fixtures: " + ", ".join(missing_fixtures))
    planned_fixtures = sorted(
        name
        for name, fixture in fixtures.items()
        if name in P8_5_REQUIRED_MIXED_FIXTURES and fixture.get("status") == "planned"
    )
    if planned_fixtures:
        fail("P8.5 lock found planned mixed fixtures: " + ", ".join(planned_fixtures))

    vc4kernel_fixture_hits = 0
    ssavc4_fixture_hits = 0
    for name in sorted(P8_5_REQUIRED_MIXED_FIXTURES):
        fixture = fixtures[name]
        fixture_dir = repo_root / fixture["path"]
        input_path = fixture_dir / "input.mlir"
        expected_path = fixture_dir / "expected.json"
        if not input_path.is_file() or not expected_path.is_file():
            fail(f"P8.5 lock missing input/expected for {name}")
        text = read_text(input_path)
        if fixture["dialect"] == "vc4kernel":
            vc4kernel_fixture_hits += 1
            if "vc4kernel.kernel" not in text:
                fail(f"P8.5 lock VC4Kernel mixed fixture lacks kernel op: {name}")
            for prefix in PRODUCER_OR_LOWER_HALF_OP_PREFIXES:
                if prefix in text:
                    fail(
                        f"P8.5 lock VC4Kernel mixed fixture {name} contains {prefix}"
                    )
        elif fixture["dialect"] == "ssavc4":
            ssavc4_fixture_hits += 1
            if "ssavc4.module" not in text:
                fail(f"P8.5 lock SSAVC4 mixed fixture lacks module op: {name}")
            if "vc4kernel." in text or "\nvc4." in text or " vc4." in text:
                fail(f"P8.5 lock SSAVC4 mixed fixture has forbidden source dialect: {name}")
        else:
            fail(f"P8.5 lock invalid mixed fixture dialect for {name}")

    missing_isolated = []
    vc4kernel_run_root = repo_root / "compiler/test/CodeGen/VC4Kernel/Hardware/Run"
    for name in sorted(P8_5_REPRESENTATIVE_ISOLATED_FIXTURES):
        if not (vc4kernel_run_root / name / "input.mlir").is_file():
            missing_isolated.append(name)
    if missing_isolated:
        fail(
            "P8.5 lock expected representative isolated fixtures retained: "
            + ", ".join(missing_isolated)
        )

    docs = read_text(repo_root / P8_5_MIXED_POLICY_DOC)
    for token in [
        "isolated fixtures remain",
        "Routine final acceptance does not run every isolated historical hardware",
        "Run the isolated fixture band",
        "P9-P13 must add mixed coverage",
    ]:
        if token not in docs:
            fail(f"P8.5 lock expected policy docs to contain: {token}")

    p9_staged_surface_tokens = {
        "vc4kernel.fragment_pack": "p9_fragment_pack",
        "vc4kernel.fragment_unpack": "p9_fragment_unpack",
    }
    p9_staged_statuses = {
        "implemented_pending_hardware",
        "hardware_proven_pending_final_acceptance",
        "accepted",
    }
    future_phase_surface_hits = []
    source_roots = [
        Path("compiler/include/vc4/Dialect/VC4Kernel"),
        Path("compiler/lib/Dialect/VC4Kernel"),
        Path("compiler/lib/Conversion/VC4KernelToSSAVC4"),
    ]
    for root in source_roots:
        for path in iter_text_files(repo_root / root, repo_root):
            text = read_text(path)
            for token in [
                "vc4kernel.fragment_pack",
                "vc4kernel.fragment_unpack",
                "vc4kernel.pack",
                "vc4kernel.unpack",
                "vc4kernel.sfu",
                "vc4kernel.fragment_sfu",
                "vc4kernel.dynamic_rotate",
                "vc4kernel.fragment_rotate_dynamic",
                "vc4kernel.dynamic_vpm_coord",
                "vc4kernel.vector",
                "vc4kernel.triton",
            ]:
                if token in text:
                    feature_id = p9_staged_surface_tokens.get(token)
                    if feature_id:
                        feature = features.get(feature_id, {})
                        if feature.get("phase") == "P9" and feature.get(
                            "current_status"
                        ) in p9_staged_statuses:
                            continue
                    future_phase_surface_hits.append((path, token))
    if future_phase_surface_hits:
        details = "; ".join(
            f"{rel(path, repo_root)}:{token}"
            for path, token in future_phase_surface_hits[:20]
        )
        fail(f"P8.5 lock found future phase surface tokens in active source: {details}")

    counts.update(
        {
            "mixed_checker_lock": 1,
            "mixed_fixtures": len(P8_5_REQUIRED_MIXED_FIXTURES),
            "mixed_vc4kernel_fixtures": vc4kernel_fixture_hits,
            "mixed_ssavc4_fixtures": ssavc4_fixture_hits,
            "isolated_fixture_retention_checks": len(P8_5_REPRESENTATIVE_ISOLATED_FIXTURES),
            "missing_isolated_fixtures": 0,
            "future_phase_surface_hits": 0,
        }
    )
    return counts


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
        "p4-general-reduce-lock",
        "p5-scalar-arith-lock",
        "p6-memory-policy-targeted",
        "p6-memory-policy-lock",
        "p7-tmu-safe-load-lock",
        "p8-vdw-store-policy-lock",
        "p8-5-mixed-acceptance-lock",
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
        "p4-general-reduce-lock",
        "p5-scalar-arith-lock",
        "p6-memory-policy-targeted",
        "p6-memory-policy-lock",
        "p8-vdw-store-policy-lock",
        "p8-5-mixed-acceptance-lock",
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
        in {
            "p1-general-alu-lock",
            "p2-bitcast-const-lock",
            "p3-general-cmp-lock",
            "p4-general-reduce-lock",
            "p5-scalar-arith-lock",
            "p6-memory-policy-targeted",
            "p6-memory-policy-lock",
            "p7-tmu-safe-load-lock",
            "p8-vdw-store-policy-lock",
            "p8-5-mixed-acceptance-lock",
        }
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
    p4_lock_counts = (
        audit_p4_general_reduce_lock(repo_root, matrix)
        if args.mode == "p4-general-reduce-lock"
        else {}
    )
    p5_lock_counts = (
        audit_p5_scalar_arith_lock(repo_root, matrix)
        if args.mode == "p5-scalar-arith-lock"
        else {}
    )
    p6_targeted_counts = (
        audit_p6_memory_policy_targeted(repo_root, matrix)
        if args.mode == "p6-memory-policy-targeted"
        else {}
    )
    p6_lock_counts = (
        audit_p6_memory_policy_lock(repo_root, matrix)
        if args.mode == "p6-memory-policy-lock"
        else {}
    )
    p7_lock_counts = (
        audit_p7_tmu_safe_load_lock(repo_root, matrix)
        if args.mode == "p7-tmu-safe-load-lock"
        else {}
    )
    p8_lock_counts = (
        audit_p8_vdw_store_policy_lock(repo_root, matrix)
        if args.mode == "p8-vdw-store-policy-lock"
        else {}
    )
    p8_5_lock_counts = (
        audit_p8_5_mixed_acceptance_lock(repo_root, matrix)
        if args.mode == "p8-5-mixed-acceptance-lock"
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
    if p4_lock_counts:
        print(f"p4_general_reduce_lock: {format_counts(p4_lock_counts)}")
    if p5_lock_counts:
        print(f"p5_scalar_arith_lock: {format_counts(p5_lock_counts)}")
    if p6_targeted_counts:
        print(f"p6_memory_policy_targeted: {format_counts(p6_targeted_counts)}")
    if p6_lock_counts:
        print(f"p6_memory_policy_lock: {format_counts(p6_lock_counts)}")
    if p7_lock_counts:
        print(f"p7_tmu_safe_load_lock: {format_counts(p7_lock_counts)}")
    if p8_lock_counts:
        print(f"p8_vdw_store_policy_lock: {format_counts(p8_lock_counts)}")
    if p8_5_lock_counts:
        print(f"p8_5_mixed_acceptance_lock: {format_counts(p8_5_lock_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
