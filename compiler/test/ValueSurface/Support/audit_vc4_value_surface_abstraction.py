#!/usr/bin/env python3
"""Audit the Phase 3.5 VC4 value-surface vector abstraction boundary."""

import argparse
import importlib.util
import json
import pathlib
import re
import sys


REQUIRED_EQUIVALENTS = {
    "value.vector.fixed_rank1.width16_fragment_carriers": (
        "fixed_vector_types_surface",
        "accept",
        "lowerable_v1",
    ),
    "value.vector.fixed_rank1.non16_staged_split": (
        "fixed_vector_types_surface",
        "accept",
        "staged_split_required",
    ),
    "value.vector.fixed_rank2.staged_tile_contract": (
        "fixed_vector_types_surface",
        "accept",
        "staged_tile_or_contract_required",
    ),
    "value.vector.scalable_reject": (
        "unsupported_scalable_vectors_and_unranked_memrefs",
        "reject",
        "rejected",
    ),
    "value.vector.unsupported_element_reject": (
        "unsupported_scalable_vectors_and_unranked_memrefs",
        "reject",
        "rejected",
    ),
    "value.tensor_linalg.initial_reject": (
        "forbidden_producer_dialects",
        "reject",
        "rejected",
    ),
    "value.vector.f16_storage_not_native_arith": (
        "staged_subword_f16_storage",
        "accept",
        "staged_storage_only",
    ),
}

FORBIDDEN_VC4VALUE_TOKENS = [
    "load",
    "store",
    "tile",
    "vpm",
    "tmu",
    "vdr",
    "vdw",
    "fragment",
    "lane",
    "barrier",
    "warp",
    "thread",
]

TARGET_OR_LOWER_HALF_TOKENS = [
    "vc4kernel.",
    "ssavc4.",
    '"vc4.',
    " vc4.",
]

ALLOWED_VECTOR16_CONTEXTS = [
    "phase 5 v1 lowerable subset",
    "phase 5 v1 lowerable",
    "fragment carrier",
    "vc4kernel carrier",
    "carrier types",
    "fragment-normal form",
    "not a wider vc4kernel fragment",
    "not the global value-layer",
    "not the global value layer",
    "not the global value-layer type limit",
    "not the global value-layer shape limit",
]


def fail(message: str) -> None:
    print(f"FAIL VC4 value surface abstraction audit: {message}", file=sys.stderr)
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


def feature_by_id(matrix: dict) -> dict[str, dict]:
    return {row["feature_id"]: row for row in matrix["features"]}


def lowering_behavior_for(row: dict, equivalent_id: str):
    behavior = row.get("phase5_lowering_behavior")
    if isinstance(behavior, dict):
        return behavior.get(equivalent_id)
    return behavior


def validate_abstraction_matrix(matrix: dict) -> None:
    rows = feature_by_id(matrix)
    for equivalent_id, (feature_id, surface_behavior, lowering_behavior) in (
        REQUIRED_EQUIVALENTS.items()
    ):
        row = rows.get(feature_id)
        if row is None:
            fail(f"missing matrix row equivalent owner {feature_id}")
        equivalents = row.get("phase3_5_equivalent_feature_ids")
        if not isinstance(equivalents, list) or equivalent_id not in equivalents:
            fail(f"{feature_id} missing abstraction equivalent {equivalent_id}")
        if row.get("surface_verifier_behavior") != surface_behavior:
            fail(f"{feature_id} has wrong surface_verifier_behavior")
        if lowering_behavior_for(row, equivalent_id) != lowering_behavior:
            fail(f"{feature_id} has wrong phase5_lowering_behavior for {equivalent_id}")

    fixed = rows["fixed_vector_types_surface"]
    fixed_text = json.dumps(fixed, sort_keys=True).lower()
    for equivalent_id in [
        "value.vector.fixed_rank1.non16_staged_split",
        "value.vector.fixed_rank2.staged_tile_contract",
    ]:
        if fixed.get("status") == "deterministic_reject_surface_policy":
            fail(f"{equivalent_id} is incorrectly classified as deterministic reject")
    if "native f16 arithmetic is accepted" in json.dumps(
        rows["staged_subword_f16_storage"], sort_keys=True
    ).lower():
        fail("f16 storage row claims native f16 arithmetic")
    if "not the global value-layer type limit" not in fixed_text:
        fail("fixed vector row does not distinguish vector<16> from global value limit")


def is_valid_value_surface_test(path: pathlib.Path) -> bool:
    name = path.name
    if name.startswith("invalid-") or "invalid" in name or "reject" in name:
        return False
    return name.endswith((".mlir", ".test")) and name.startswith("value-surface-")


def scan_valid_tests(repo_root: pathlib.Path) -> list[str]:
    hits = []
    test_dir = repo_root / "compiler/test/ValueSurface"
    for path in sorted(test_dir.glob("*")):
        if not path.is_file() or not is_valid_value_surface_test(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in TARGET_OR_LOWER_HALF_TOKENS:
            if token in text:
                hits.append(f"{path}:{token}")
    return hits


def scan_vector16_only_claims(repo_root: pathlib.Path) -> list[str]:
    hits = []
    roots = [
        repo_root / "compiler/docs",
        repo_root / "compiler/test/ValueSurface",
        repo_root / "compiler/lib/Transforms/ValueSurface",
    ]
    pattern = re.compile(r"vector<16(?:xT)?>.*only|only.*vector<16(?:xT)?>|simd-16.*only", re.I)
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if "Support" in path.parts:
                continue
            if path.name.startswith("invalid-") or "invalid" in path.name:
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                lower = line.lower()
                if not pattern.search(line):
                    continue
                if any(context in lower for context in ALLOWED_VECTOR16_CONTEXTS):
                    continue
                hits.append(f"{path}:{lineno}:{line.strip()}")
    return hits


def scan_readiness(repo_root: pathlib.Path) -> tuple[list[str], list[str]]:
    triton_hits = []
    lowering_hits = []
    roots = [repo_root / "compiler/docs", repo_root / "compiler/test"]
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if "Support" in path.parts:
                continue
            if path.name.startswith("invalid-") or "invalid" in path.name:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "READY_FOR_TRITON=YES" in text:
                triton_hits.append(str(path))
            if path.name in {
                "vc4_vector_triton_phase3_value_surface_lock.md",
                "vc4_vector_triton_phase3_5_value_surface_lock.md",
                "vc4_value_surface_abstraction_policy.md",
            } and "READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=YES" in text:
                lowering_hits.append(str(path))
    return triton_hits, lowering_hits


def scan_vc4value_scope(repo_root: pathlib.Path) -> list[str]:
    hits = []
    roots = [
        repo_root / "compiler/include/vc4/Dialect/VC4Value",
        repo_root / "compiler/lib/Dialect/VC4Value",
    ]
    op_def_pattern = re.compile(r"def\s+VC4Value_([A-Za-z0-9_]+)Op")
    mnemonic_pattern = re.compile(r"VC4Value_[A-Za-z0-9_]*Op<\"([^\"]+)\"")
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".td", ".cpp", ".h"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in mnemonic_pattern.finditer(text):
                mnemonic = match.group(1).lower()
                for token in FORBIDDEN_VC4VALUE_TOKENS:
                    if token in mnemonic:
                        hits.append(f"{path}:mnemonic:{mnemonic}")
            for match in op_def_pattern.finditer(text):
                op_name = match.group(1).lower()
                for token in FORBIDDEN_VC4VALUE_TOKENS:
                    if token in op_name:
                        hits.append(f"{path}:opdef:{op_name}")
            for token in FORBIDDEN_VC4VALUE_TOKENS:
                if f"vc4value.{token}" in text:
                    hits.append(f"{path}:vc4value.{token}")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    if args.mode != "phase3_5":
        fail(f"unsupported mode {args.mode!r}")

    repo_root = pathlib.Path(args.repo_root).resolve()
    matrix_path = pathlib.Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = repo_root / matrix_path

    checker = load_matrix_checker()
    matrix = checker.validate_matrix(str(matrix_path), "phase3_5")
    validate_abstraction_matrix(matrix)

    vector16_only_hits = scan_vector16_only_claims(repo_root)
    valid_target_hits = scan_valid_tests(repo_root)
    triton_hits, lowering_hits = scan_readiness(repo_root)
    vc4value_scope_hits = scan_vc4value_scope(repo_root)

    if vector16_only_hits:
        fail("vector<16>-only claims: " + "; ".join(vector16_only_hits))
    if valid_target_hits:
        fail("target/lower-half mentions in valid tests: " + "; ".join(valid_target_hits))
    if triton_hits:
        fail("READY_FOR_TRITON=YES hits: " + "; ".join(triton_hits))
    if lowering_hits:
        fail("READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=YES hits: " + "; ".join(lowering_hits))
    if vc4value_scope_hits:
        fail("vc4value scope creep hits: " + "; ".join(vc4value_scope_hits))

    print(f"matrix_features={len(matrix['features'])}")
    print(f"abstraction_equivalents={len(REQUIRED_EQUIVALENTS)}")
    print(f"vector16_only_hits={len(vector16_only_hits)}")
    print(f"valid_target_hits={len(valid_target_hits)}")
    print(f"readiness_yes_hits={len(triton_hits) + len(lowering_hits)}")
    print(f"vc4value_scope_creep_hits={len(vc4value_scope_hits)}")
    print("PASS VC4 value surface abstraction audit: mode=phase3_5")


if __name__ == "__main__":
    main()
