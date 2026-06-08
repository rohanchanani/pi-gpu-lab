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
}


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


def require_file(path: pathlib.Path, label: str) -> None:
    if not path.exists():
        fail(f"{label} missing: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    if args.mode != "phase3-lock":
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
    readiness_yes_hits = scan_readiness_yes(repo_root)

    if forbidden_valid_hits:
        fail("forbidden dialects in valid tests: " + ", ".join(forbidden_valid_hits))
    if vc4value_scope_hits:
        fail("vc4value scope creep hits: " + ", ".join(vc4value_scope_hits))
    if readiness_yes_hits:
        fail("readiness YES hits: " + ", ".join(readiness_yes_hits))

    print(f"matrix_features={matrix_features}")
    print(f"valid_test_files={len(valid_files)}")
    print(f"invalid_test_files={len(invalid_files)}")
    print(f"forbidden_valid_test_hits={len(forbidden_valid_hits)}")
    print(f"vc4value_scope_creep_hits={len(vc4value_scope_hits)}")
    print(f"readiness_yes_hits={len(readiness_yes_hits)}")
    print("PASS VC4 value surface audit: mode=phase3-lock")


if __name__ == "__main__":
    main()
