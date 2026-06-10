#!/usr/bin/env python3
"""Audit the Phase 3 VC4 value-surface verifier contract."""

import argparse
import importlib.util
import json
import pathlib
import re
import sys


FORBIDDEN_VALID_TEST_TOKENS = [
    "vc4kernel.",
    "ssavc4.",
    "vc4.",
    "tt.",
    "ttg.",
    "gpu.",
    "linalg.",
    "nvgpu.",
    "nvvm.",
    "rocdl.",
    "spirv.",
    "iree.",
    "stablehlo.",
    "mhlo.",
    "tensor.",
]

FORBIDDEN_VC4VALUE_SCOPE_TOKENS = [
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
    "barrier",
]

COVERAGE_REQUIREMENTS = {
    "value_to_vc4kernel_lowering_absent_guard": "value-surface-audit.test",
    "triton_ttir_direct_ingestion_absent_guard": "value-surface-audit.test",
    "forbidden_producer_dialects": "invalid-value-surface-producer-dialects.mlir",
    "forbidden_target_dialects": "invalid-value-surface-target-dialects.mlir",
    "forbidden_vc4value_scope_creep": "invalid-value-surface-vc4value-scope-creep.mlir",
    "forbidden_memref_side_effect_ops": "invalid-value-surface-memory-side-effects.mlir",
    "forbidden_sparse_store_ops": "invalid-value-surface-sparse-store-ops.mlir",
    "unsupported_scalable_vectors_and_unranked_memrefs": "invalid-value-surface-type-guardrails.mlir",
    "phase8_cf_if_valid": "value-surface-cf-if-valid.mlir",
    "phase8_cf_loop_valid": "value-surface-cf-loop-valid.mlir",
    "phase8_scf_boundary_valid": "value-surface-scf-boundary-valid.mlir",
    "phase8r_scf_while_valid": "value-surface-scf-while-valid.mlir",
    "phase8r_nested_structured_cf_valid": "value-surface-nested-structured-cf-valid.mlir",
    "phase8r_vector_branch_condition_invalid": "value-surface-vector-branch-condition-invalid.mlir",
    "phase8r_scf_parallel_staged_invalid": "value-surface-scf-parallel-or-forall-staged-invalid.mlir",
    "phase8r_scf_index_switch_policy": "value-surface-scf-index-switch-policy.mlir",
    "phase8_cf_policy_invalid": "invalid-value-surface-phase8-cf-policy.mlir",
}

PHASE4_VALID_CORPUS = [
    "value-abi-basic-launch-valid.mlir",
    "value-abi-grid-rank-3-valid.mlir",
    "value-abi-public-scalar-args-valid.mlir",
    "value-abi-memref-element-types-rank1-valid.mlir",
    "value-abi-memref-element-types-rank2-valid.mlir",
    "value-abi-memref-dim-metadata-valid.mlir",
    "value-abi-static-memref-no-shape-args-valid.mlir",
    "value-abi-stride-args-metadata-valid.mlir",
    "value-abi-vector-values-not-public-args-valid.mlir",
]

PHASE4_INVALID_CORPUS = {
    "missing kernel grid_rank": (
        "invalid-value-abi-kernel-wrapper-and-arg-attrs.mlir",
        "missing_grid_rank",
    ),
    "bad grid_rank": (
        "invalid-value-abi-kernel-wrapper-and-arg-attrs.mlir",
        "grid_rank_four",
    ),
    "axis out of range": ("invalid-value-abi-memref-abi.mlir", "axis = 1 : i32"),
    "missing arg_name": (
        "invalid-value-abi-kernel-wrapper-and-arg-attrs.mlir",
        "missing_arg_name",
    ),
    "duplicate arg_name": (
        "invalid-value-abi-kernel-wrapper-and-arg-attrs.mlir",
        "duplicate_arg_name",
    ),
    "bad arg_name": (
        "invalid-value-abi-kernel-wrapper-and-arg-attrs.mlir",
        "bad_arg_name",
    ),
    "reserved arg_name": (
        "invalid-value-abi-kernel-wrapper-and-arg-attrs.mlir",
        "reserved_arg_name",
    ),
    "missing direction": (
        "invalid-value-abi-kernel-wrapper-and-arg-attrs.mlir",
        "missing_direction",
    ),
    "bad direction": (
        "invalid-value-abi-kernel-wrapper-and-arg-attrs.mlir",
        "bad_direction",
    ),
    "scalar direction": (
        "invalid-value-abi-kernel-wrapper-and-arg-attrs.mlir",
        "scalar_direction",
    ),
    "bad scalar_role": (
        "invalid-value-abi-kernel-wrapper-and-arg-attrs.mlir",
        "bad_scalar_role",
    ),
    "vector public arg": (
        "invalid-value-abi-kernel-wrapper-and-arg-attrs.mlir",
        "vector_arg",
    ),
    "tensor public arg": (
        "invalid-value-abi-kernel-wrapper-and-arg-attrs.mlir",
        "tensor_arg",
    ),
    "unranked memref": ("invalid-value-abi-memref-abi.mlir", "memref<*xf32>"),
    "rank0 memref": ("invalid-value-abi-memref-abi.mlir", "@rank0"),
    "rank3 memref": ("invalid-value-abi-memref-abi.mlir", "@rank3"),
    "missing global memspace": ("invalid-value-abi-memref-abi.mlir", "@default_memspace"),
    "wrong memspace": ("invalid-value-abi-memref-abi.mlir", "@wrong_memspace"),
    "memref i1 reject": ("invalid-value-abi-memref-abi.mlir", "@memref_i1"),
    "memref i64 reject": ("invalid-value-abi-memref-abi.mlir", "@memref_i64"),
    "memref f64 reject": ("invalid-value-abi-memref-abi.mlir", "@memref_f64"),
    "memref bf16 reject": ("invalid-value-abi-memref-abi.mlir", "@memref_bf16"),
    "memref index reject": ("invalid-value-abi-memref-abi.mlir", "@memref_index"),
    "missing shape_args": ("invalid-value-abi-memref-abi.mlir", "@missing_shape_args"),
    "shape_args length mismatch": ("invalid-value-abi-memref-abi.mlir", "@shape_len"),
    "shape_args nonexistent": ("invalid-value-abi-memref-abi.mlir", "@shape_missing_arg"),
    "shape_args wrong type": ("invalid-value-abi-memref-abi.mlir", "@shape_f32_arg"),
    "shape_args duplicate": ("invalid-value-abi-memref-abi.mlir", "@shape_duplicate"),
    "shape_args malformed": ("invalid-value-abi-memref-abi.mlir", "@shape_malformed"),
    "stride_args nonexistent": ("invalid-value-abi-memref-abi.mlir", "@stride_missing_arg"),
    "stride_args malformed": ("invalid-value-abi-memref-abi.mlir", "@stride_malformed"),
    "stride_args wrong type": ("invalid-value-abi-memref-abi.mlir", "@stride_f32_arg"),
    "memref.load reject": ("invalid-value-abi-memref-abi.mlir", "memref.load"),
    "memref.store reject": ("invalid-value-abi-memref-abi.mlir", "memref.store"),
    "vc4value hardware op names": (
        "invalid-value-surface-vc4value-scope-creep.mlir",
        "vc4value.load",
    ),
}

PHASE4_ALLOWED_SCOPED_LOWERING = (
    "READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=YES_FOR_PHASE5_ELEMENTWISE_V1"
)


def fail(message: str) -> None:
    print(f"FAIL VC4 value surface audit: {message}", file=sys.stderr)
    sys.exit(1)


def load_matrix_checker():
    here = pathlib.Path(__file__).resolve().parent
    checker_path = here / "check_vc4_value_surface_matrix.py"
    spec = importlib.util.spec_from_file_location("check_vc4_value_surface_matrix", checker_path)
    if spec is None or spec.loader is None:
        fail("could not import matrix checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repo_path(repo_root: pathlib.Path, rel: str) -> pathlib.Path:
    return repo_root / rel


def is_valid_value_surface_test(path: pathlib.Path) -> bool:
    name = path.name
    if name.startswith("invalid-") or "reject" in name or "invalid" in name:
        return False
    return (
        name.startswith("value-surface-")
        or name == "verify-value-surface-basic-valid.mlir"
        or name == "verify-value-surface-allowed-standard-ops.mlir"
    )


def is_invalid_value_surface_test(path: pathlib.Path) -> bool:
    return (
        path.name.startswith("invalid-")
        or "reject" in path.name
        or "invalid" in path.name
    )


def extract_vc4value_op_mnemonics(ops_td: pathlib.Path) -> list[str]:
    text = ops_td.read_text(encoding="utf-8")
    return re.findall(r"def\s+VC4Value_\w+Op\s*:\s*VC4Value_\w*Op<\"([^\"]+)\"", text)


def scan_vc4value_scope(repo_root: pathlib.Path) -> list[str]:
    hits = []
    ops_td = repo_path(repo_root, "compiler/include/vc4/Dialect/VC4Value/IR/VC4ValueOps.td")
    if not ops_td.exists():
        fail("VC4ValueOps.td missing")
    for mnemonic in extract_vc4value_op_mnemonics(ops_td):
        for token in FORBIDDEN_VC4VALUE_SCOPE_TOKENS:
            if token in mnemonic:
                hits.append(f"{ops_td}:{mnemonic}:{token}")
    return hits


def scan_valid_tests(test_dir: pathlib.Path) -> tuple[list[pathlib.Path], list[pathlib.Path], list[str]]:
    valid_files = []
    invalid_files = []
    forbidden_hits = []
    for path in sorted(test_dir.glob("*.mlir")):
        if is_valid_value_surface_test(path):
            valid_files.append(path)
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN_VALID_TEST_TOKENS:
                if token in text:
                    forbidden_hits.append(f"{path}:{token}")
        if is_invalid_value_surface_test(path):
            invalid_files.append(path)
    return valid_files, invalid_files, forbidden_hits


def scan_readiness_yes(repo_root: pathlib.Path) -> list[str]:
    hits = []
    for rel in [
        "compiler/docs/vc4_value_surface_verifier.md",
        "compiler/docs/vc4_value_surface_support_matrix.json",
    ]:
        path = repo_path(repo_root, rel)
        text = path.read_text(encoding="utf-8")
        for needle in [
            "READY_FOR_TRITON=YES",
            "READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=YES",
        ]:
            if needle in text:
                hits.append(f"{path}:{needle}")
    return hits


def scan_phase4_readiness_yes(repo_root: pathlib.Path) -> list[str]:
    hits = []
    for root in [repo_root / "compiler/docs", repo_root / "compiler/test"]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or "Support" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "READY_FOR_TRITON=YES" in text:
                hits.append(f"{path}:READY_FOR_TRITON=YES")
            broad_lowering_text = text.replace(PHASE4_ALLOWED_SCOPED_LOWERING, "")
            if "READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=YES" in broad_lowering_text:
                hits.append(f"{path}:READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=YES")
    return hits


def scan_phase4_doc_claims(repo_root: pathlib.Path) -> list[str]:
    hits = []
    docs = [
        repo_path(repo_root, "compiler/docs/vc4_value_kernel_abi.md"),
        repo_path(repo_root, "compiler/docs/vc4_value_surface_verifier.md"),
        repo_path(repo_root, "compiler/docs/vc4_value_surface_support_matrix.json"),
    ]
    hardware_names = ["TMU", "VDR", "VPM", "VDW"]
    for path in docs:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "#vc4value.global" in line and "select" in line.lower():
                if any(name in line for name in hardware_names) and "does not" not in line.lower():
                    hits.append(f"{path}:{lineno}:#vc4value.global hardware selection")
            lowered = line.lower()
            for phrase in [
                "uses hidden memref descriptor",
                "hidden memref descriptors are accepted",
                "hidden descriptor abi is accepted",
                "all phase 4 memref types are phase 5 lowerable",
                "native f16 arithmetic is accepted",
                "native f16 arithmetic is lowerable",
            ]:
                if phrase in lowered:
                    hits.append(f"{path}:{lineno}:{phrase}")
            if "direct vc4kerneltovc4" in lowered:
                allowed = ["no direct", "not", "forbidden", "non-goal", "do not"]
                if not any(token in lowered for token in allowed):
                    hits.append(f"{path}:{lineno}:direct VC4KernelToVC4 accepted path")
            if "vc4tile" in lowered:
                allowed = ["no vc4tile", "not", "forbidden", "retired", "resurrect"]
                if not any(token in lowered for token in allowed):
                    hits.append(f"{path}:{lineno}:VC4Tile current path")
    return hits


def scan_phase8_cf_policy(repo_root: pathlib.Path) -> list[str]:
    hits = []
    docs = [
        repo_path(repo_root, "compiler/docs/vc4_value_surface_spec.md"),
        repo_path(repo_root, "compiler/docs/vc4_value_to_vc4kernel_planning.md"),
        repo_path(repo_root, "compiler/docs/vc4_value_surface_support_matrix.json"),
    ]
    required_phrases = [
        "Phase 8 is a value-layer control-flow phase",
        "scf-to-cf",
        "PHASE8R_CF_COMPLETENESS_SCOPE=ACTIVE",
        "SCF_WHILE_VALUE_TARGET=SUPPORT_NOW",
        "NESTED_STRUCTURED_CF_VALUE_TARGET=SUPPORT_NOW",
        "TL_RANGE_STYLE_LOOP_SKELETON_VALUE_TARGET=SUPPORT_NOW",
        "PERSISTENT_LOOP_SKELETON_VALUE_TARGET=SUPPORT_NOW",
        "VECTOR_BRANCH_CONDITION_POLICY=DETERMINISTIC_REJECT_AS_CFG_USE_MASKS",
        "IRREDUCIBLE_CFG_POLICY=PROBE_NOT_REQUIRED_FOR_SANE_TRITON",
        "TTIR control flow",
        "READY_FOR_TRITON remains NO",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for phrase in required_phrases:
            if phrase not in text:
                hits.append(f"{path}:missing {phrase}")
        if "READY_FOR_TRITON=YES" in text:
            hits.append(f"{path}:READY_FOR_TRITON=YES")

    matrix = json.loads(
        repo_path(repo_root, "compiler/docs/vc4_value_surface_support_matrix.json")
        .read_text(encoding="utf-8")
    )
    rows = {row.get("feature_id"): row for row in matrix.get("features", [])}
    control_row = rows.get("scf_cf_surface_boundary", {})
    behavior = control_row.get("phase8_control_flow_behavior", {})
    expected = {
        "cf.br": "accepted_executable_phase8_pending_hardware",
        "cf.cond_br": "accepted_executable_phase8_pending_hardware",
        "scf.if": "surface_admissible_canonicalization_required",
        "scf.for": "surface_admissible_canonicalization_required",
        "scf.while": "surface_admissible_canonicalization_required_phase8r",
        "nested_structured_cf": "support_now_phase8r",
        "tl_range_style_loop_skeleton": "support_now_value_level_phase8r",
        "persistent_loop_skeleton": "support_now_value_level_phase8r",
        "ttir_control_flow": "staged_future_ttir_import",
        "vector_valued_branch_conditions": "deterministic_reject_as_cfg_use_masks",
        "memref_block_args": "deterministic_reject_until_proven",
        "cf.switch": "lowerable_phase8rd_scalar_chain",
        "scf.index_switch": "canonicalization_required_phase8rd_to_cf_switch",
        "multi_exit_reducible_loops": "lowerable_phase8rd_static",
        "scf.parallel": "staged_reject_parallel_semantics",
        "scf.forall": "staged_reject_parallel_semantics",
        "scf.reduce": "staged_reject_reduction_semantics",
        "irreducible_cfg": "probe_not_required_for_sane_triton",
    }
    for key, value in expected.items():
        if behavior.get(key) != value:
            hits.append(f"matrix scf/cf behavior {key}={behavior.get(key)!r}")

    ops_td = repo_path(repo_root, "compiler/include/vc4/Dialect/VC4Value/IR/VC4ValueOps.td")
    vc4value_ops = extract_vc4value_op_mnemonics(ops_td)
    for control_token in ["br", "cond_br", "if", "for", "yield", "switch"]:
        if any(control_token == op or control_token in op for op in vc4value_ops):
            hits.append(f"vc4value op owns control-flow token {control_token}")

    lock_doc = repo_path(repo_root, "compiler/docs/vc4_vector_triton_phase8_control_flow_lock.md")
    lock_text = lock_doc.read_text(encoding="utf-8")
    for phrase in [
        "QPU control flow is scalar/coherent across SIMD lanes",
        "Per-lane control divergence is not accepted as branch CFG",
        "scf.while is in scope now through upstream scf-to-cf",
        "tl.range-style loop skeleton",
        "persistent-loop skeleton",
        "scf.parallel",
        "scf.forall",
        "scf.reduce",
        "cf.switch",
        "scf.index_switch",
        "PHASE8R_MULTI_EXIT_REDUCIBLE_LOOP_POLICY=LOCKED",
        "verified VC4Kernel contains no raw scf",
    ]:
        if phrase not in lock_text:
            hits.append(f"phase8 lock doc missing Phase 8R phrase: {phrase}")

    return hits


def scan_valid_abi_tests(test_dir: pathlib.Path) -> list[str]:
    hits = []
    signature_re = re.compile(
        r"func\.func\s+@[A-Za-z0-9_.$-]+\s*\((.*?)\)\s*attributes\s*\{[^}]*vc4value\.kernel",
        re.S,
    )
    for path in sorted(test_dir.glob("value-abi*.mlir")):
        text = path.read_text(encoding="utf-8")
        for token in ["memref.load", "memref.store"]:
            if token in text:
                hits.append(f"{path}:{token} in valid ABI test")
        for match in signature_re.finditer(text):
            args = match.group(1)
            if "vector<" in args:
                hits.append(f"{path}:public vector argument in valid ABI test")
            if "tensor<" in args:
                hits.append(f"{path}:public tensor argument in valid ABI test")
    return hits


def validate_phase4_corpus(test_dir: pathlib.Path) -> None:
    for filename in PHASE4_VALID_CORPUS:
        path = test_dir / filename
        require_file(path, "Phase 4 valid ABI test")
        text = path.read_text(encoding="utf-8")
        if "--vc4-verify-value-surface" not in text:
            fail(f"{filename} does not run --vc4-verify-value-surface")
    for label, (filename, needle) in PHASE4_INVALID_CORPUS.items():
        path = test_dir / filename
        require_file(path, f"Phase 4 invalid ABI test for {label}")
        text = path.read_text(encoding="utf-8")
        if needle not in text:
            fail(f"{filename} missing Phase 4 invalid ABI coverage for {label}: {needle}")


def require_file(path: pathlib.Path, label: str) -> None:
    if not path.exists():
        fail(f"{label} missing: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    if args.mode not in {"phase3-lock", "phase4-abi-lock"}:
        fail(f"unsupported mode {args.mode!r}")

    repo_root = pathlib.Path(args.repo_root).resolve()
    matrix_path = pathlib.Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = repo_root / matrix_path

    checker = load_matrix_checker()
    matrix = checker.validate_matrix(str(matrix_path), args.mode)
    matrix_features = len(matrix["features"])

    verifier_doc = repo_path(repo_root, "compiler/docs/vc4_value_surface_verifier.md")
    verifier_pass = repo_path(
        repo_root, "compiler/lib/Transforms/ValueSurface/ValueSurfaceVerification.cpp"
    )
    require_file(verifier_doc, "verifier doc")
    require_file(verifier_pass, "verifier pass")

    doc_text = verifier_doc.read_text(encoding="utf-8")
    normalized_doc_text = " ".join(doc_text.split())
    if (
        "verifies value-surface admissibility, not current lowerability"
        not in normalized_doc_text
    ):
        fail(
            "verifier doc missing phrase: verifies value-surface "
            "admissibility, not current lowerability"
        )
    for phrase in [
        "READY_FOR_TRITON=NO",
        "READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=NO",
    ]:
        if phrase not in doc_text:
            fail(f"verifier doc missing phrase: {phrase}")

    test_dir = repo_path(repo_root, "compiler/test/ValueSurface")
    valid_files, invalid_files, forbidden_valid_hits = scan_valid_tests(test_dir)
    if not valid_files:
        fail("no valid value-surface tests found")
    if not invalid_files:
        fail("no invalid value-surface tests found")

    for feature_id, filename in COVERAGE_REQUIREMENTS.items():
        path = test_dir / filename
        if not path.exists():
            fail(f"missing diagnostic coverage for {feature_id}: {filename}")
        text = path.read_text(encoding="utf-8")
        if feature_id in {
            "value_to_vc4kernel_lowering_absent_guard",
            "triton_ttir_direct_ingestion_absent_guard",
        }:
            if "audit_vc4_value_surface.py" not in text:
                fail(f"{filename} does not run the value-surface audit")
        elif "--vc4-verify-value-surface" not in text:
            fail(f"{filename} does not run --vc4-verify-value-surface")

    lit_mentions = 0
    for path in sorted(test_dir.glob("*")):
        if path.suffix in {".mlir", ".test"}:
            if "--vc4-verify-value-surface" in path.read_text(encoding="utf-8"):
                lit_mentions += 1
    if lit_mentions == 0:
        fail("lit tests do not mention --vc4-verify-value-surface")

    vc4value_scope_hits = scan_vc4value_scope(repo_root)
    readiness_yes_hits = (
        scan_phase4_readiness_yes(repo_root)
        if args.mode == "phase4-abi-lock"
        else scan_readiness_yes(repo_root)
    )
    phase4_doc_hits = []
    phase4_valid_abi_hits = []
    phase8_cf_policy_hits = scan_phase8_cf_policy(repo_root)
    if args.mode == "phase4-abi-lock":
        validate_phase4_corpus(test_dir)
        phase4_doc_hits = scan_phase4_doc_claims(repo_root)
        phase4_valid_abi_hits = scan_valid_abi_tests(test_dir)

    if forbidden_valid_hits:
        fail("forbidden dialects in valid tests: " + ", ".join(forbidden_valid_hits))
    if vc4value_scope_hits:
        fail("vc4value scope creep hits: " + ", ".join(vc4value_scope_hits))
    if readiness_yes_hits:
        fail("readiness YES hits: " + ", ".join(readiness_yes_hits))
    if phase4_doc_hits:
        fail("Phase 4 ABI doc claim hits: " + ", ".join(phase4_doc_hits))
    if phase4_valid_abi_hits:
        fail("Phase 4 valid ABI test hits: " + ", ".join(phase4_valid_abi_hits))
    if phase8_cf_policy_hits:
        fail("Phase 8 CF policy hits: " + ", ".join(phase8_cf_policy_hits))

    print(f"matrix_features={matrix_features}")
    print(f"valid_test_files={len(valid_files)}")
    print(f"invalid_test_files={len(invalid_files)}")
    print(f"forbidden_valid_test_hits={len(forbidden_valid_hits)}")
    print(f"vc4value_scope_creep_hits={len(vc4value_scope_hits)}")
    print(f"readiness_yes_hits={len(readiness_yes_hits)}")
    if args.mode == "phase4-abi-lock":
        print(f"phase4_doc_claim_hits={len(phase4_doc_hits)}")
        print(f"phase4_valid_abi_hits={len(phase4_valid_abi_hits)}")
        print(f"phase4_valid_corpus={len(PHASE4_VALID_CORPUS)}")
        print(f"phase4_invalid_corpus={len(PHASE4_INVALID_CORPUS)}")
    print(f"phase8_cf_policy_hits={len(phase8_cf_policy_hits)}")
    print(f"PASS VC4 value surface audit: mode={args.mode}")


if __name__ == "__main__":
    main()
