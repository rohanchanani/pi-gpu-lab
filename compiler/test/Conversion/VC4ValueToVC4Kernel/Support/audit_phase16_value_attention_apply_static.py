#!/usr/bin/env python3
"""Audit Phase 16 attention-apply static lowering scope.

This audit checks source and test contracts only. It intentionally does not
decide compiler semantics from fixture names.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    conversion = read(
        root
        / "compiler/lib/Conversion/VC4ValueToVC4Kernel/VC4ValueToVC4Kernel.cpp"
    )
    value_verifier = read(
        root / "compiler/lib/Transforms/ValueSurface/ValueSurfaceVerification.cpp"
    )
    planning = read(root / "compiler/docs/vc4_value_to_vc4kernel_planning.md")
    fixtures = read(
        root / "compiler/docs/vc4_vector_triton_phase16_attention_apply_v0_fixtures.md"
    )

    require(
        "vc4value.attention" not in conversion
        and "softmax_apply" not in conversion
        and "attention_apply_v0" not in conversion,
        "attention apply lowering must remain composition of existing planners",
    )
    require(
        "scalar memref.load is staged" in conversion,
        "scalar memref.load must remain staged in value-to-VC4Kernel conversion",
    )
    require(
        "rank-2 row-slice transfer map must project the innermost dimension"
        in conversion
        and "gather-like maps are " in conversion
        and '"staged"' in conversion,
        "non-transposed V gather/lane-varying stride must remain staged",
    )
    require(
        "vector.contract is staged" in conversion
        and "tt.dot is staged" in conversion
        and "tl.dot" not in conversion,
        "dot/contract forms must remain staged and unimplemented here",
    )
    require(
        "Public math.exp is natural exp" in conversion
        and "exp2(x * log2(e))" in conversion
        and "kLog2E" in conversion,
        "natural-exp softmax semantics must preserve Phase15B exp2 scaling",
    )
    require(
        "precomputed_transposed_v_active_1_to_16" in value_verifier,
        "value surface must carry the locked Phase16 attention metadata spelling",
    )
    require(
        "VALUE_ATTENTION_APPLY_V0_STATIC=PASS" in planning
        and "VALUE_ATTENTION_APPLY_V0_COMPOSITE=YES" in planning
        and "READY_FOR_PHASE16_5_VALUE_HARDWARE_ISOLATION=YES" in planning,
        "planning doc must record Phase16.4 static lock readiness",
    )
    require(
        "SCALAR_GLOBAL_LOAD_STAGED=YES" in fixtures
        and "NONTRANSPOSED_V_GATHER_STAGED=YES" in fixtures
        and "READY_FOR_TRITON=NO" in fixtures,
        "fixture doc must preserve staged exclusions and Triton gate",
    )

    tests = root / "compiler/test/Conversion/VC4ValueToVC4Kernel"
    required_tests = [
        "attention-apply-v0-f32-lowers.mlir",
        "attention-apply-v0-scaled-f32-lowers.mlir",
        "attention-apply-v0-f16-storage-inputs-lowers.mlir",
        "invalid-attention-apply-scalar-global-load.mlir",
        "invalid-attention-apply-nontransposed-v-gather.mlir",
        "invalid-attention-apply-k-zero-without-guard.mlir",
        "invalid-attention-apply-exact-math-no-policy.mlir",
        "invalid-attention-apply-multiblock-softmax.mlir",
    ]
    for name in required_tests:
        require((tests / name).is_file(), f"missing Phase16.4 conversion test {name}")

    print("VALUE_ATTENTION_APPLY_SOURCE_AUDIT=PASS")
    print("ATTENTION_APPLY_STANDARD_VALUE_COMPOSITE=YES")
    print("NO_NEW_VC4VALUE_ATTENTION_OP=YES")
    print("NO_SCALAR_MEMREF_LOAD_SUPPORT_INTRODUCED=YES")
    print("NONTRANSPOSED_V_GATHER_REMAINS_STAGED=YES")
    print("NATURAL_EXP_SOFTMAX_SEMANTICS_PRESERVED=YES")
    print("NO_SOURCE_NAME_PATH_FIXTURE_SPECIAL_CASES=YES")
    print("QK_SCORE_GENERATION_REMAINS_STAGED=YES")
    print("DOT_CONTRACT_REMAINS_STAGED=YES")
    print("READY_FOR_TRITON=NO")


if __name__ == "__main__":
    main()
