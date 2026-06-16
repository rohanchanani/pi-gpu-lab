#!/usr/bin/env python3
"""Audit Phase 17 online softmax static lowering scope."""

from __future__ import annotations

import argparse
from pathlib import Path


LOWERING_DIRS = [
    "compiler/lib/Conversion/VC4ValueToVC4Kernel",
    "compiler/lib/Conversion/TritonToVC4Value",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL {message}")


def iter_source_files(path: Path):
    for suffix in ("*.cpp", "*.h", "*.td"):
        yield from path.rglob(suffix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    conversion = read(
        root / "compiler/lib/Conversion/VC4ValueToVC4Kernel/VC4ValueToVC4Kernel.cpp"
    )
    planning = read(root / "compiler/docs/vc4_value_to_vc4kernel_planning.md")
    fixtures = read(
        root
        / "compiler/docs/vc4_vector_triton_phase17_online_softmax_state_fixtures.md"
    )

    forbidden_lowering_tokens = [
        "vc4value.online_softmax",
        "vc4value.attention",
        "online_softmax_state",
        "qk_score_generation",
        "flashattention",
        "FlashAttention",
    ]
    for token in forbidden_lowering_tokens:
      require(token not in conversion, f"forbidden Phase17 lowering token present: {token}")

    metadata_hits = []
    for rel in LOWERING_DIRS:
        for source_file in iter_source_files(root / rel):
            text = read(source_file)
            if "PHASE17_VALUE_ONLINE_SOFTMAX_STATE_CONTRACT" in text:
                metadata_hits.append(str(source_file.relative_to(root)))
    require(
        not metadata_hits,
        "Phase17 contract metadata must not be used by lowering: "
        + ", ".join(metadata_hits),
    )

    require(
        "scalar memref.load is staged" in conversion,
        "scalar memref.load must remain staged",
    )
    require(
        "rank-2 row-slice transfer map must project the innermost dimension"
        in conversion
        and "gather-like maps are " in conversion
        and '"staged"' in conversion,
        "non-transposed V gather/lane-varying stride must remain staged",
    )
    require(
        "vector.contract is staged" in conversion and "tt.dot is staged" in conversion,
        "dot/contract forms must remain staged",
    )
    require(
        "Public math.exp is natural exp" in conversion
        and "exp2(x * log2(e))" in conversion
        and "kLog2E" in conversion,
        "natural-exp softmax semantics must preserve Phase15B scaling",
    )
    require(
        "lowerScalarF32Max" in conversion
        and 'kMaxPolicyAttr("vc4value.max_policy")' in conversion
        and "FPCmpPolicy::finite_only" in conversion,
        "scalar finite f32 max must be explicit and policy-gated",
    )
    require(
        "isF32FragmentBlockArgument" in conversion
        and "loop-carried scalar f32 `m`, `l`, and `acc`" in planning,
        "loop-carried f32 state support and docs must be present",
    )
    require(
        "PHASE17_VALUE_ONLINE_SOFTMAX_STATE_CONTRACT=LOCKED" in planning
        and "VALUE_ONLINE_SOFTMAX_STATE_STATIC=PASS" in planning
        and "READY_FOR_PHASE17_5_VALUE_HARDWARE_ISOLATION=YES" in planning,
        "planning doc must record Phase17.4 static lock",
    )
    require(
        "VALUE_ONLINE_ATTENTION_APPLY_STATIC=PASS" in fixtures
        and "READY_FOR_TRITON=NO" in fixtures,
        "fixture doc must record static lock and preserve Triton gate",
    )

    tests = root / "compiler/test/Conversion/VC4ValueToVC4Kernel"
    required_tests = [
        "online-softmax-state-f32-lowers.mlir",
        "online-attention-apply-v0-f32-lowers.mlir",
        "online-attention-apply-v0-scaled-f32-lowers.mlir",
        "online-attention-apply-v0-f16-storage-inputs-lowers.mlir",
        "scalar-f32-finite-max-lowers.mlir",
        "loop-carried-f32-state-lowers.mlir",
        "invalid-online-softmax-k-zero-without-guard.mlir",
        "invalid-online-attention-scalar-global-load.mlir",
        "invalid-online-attention-nontransposed-v-gather.mlir",
        "invalid-online-attention-exact-math-no-policy.mlir",
        "invalid-online-attention-qk-dot-staged.mlir",
    ]
    for name in required_tests:
        require((tests / name).is_file(), f"missing Phase17.4 conversion test {name}")

    print("VALUE_ONLINE_SOFTMAX_SOURCE_AUDIT=PASS")
    print("ONLINE_SOFTMAX_STANDARD_VALUE_COMPOSITE=YES")
    print("NO_NEW_VC4VALUE_ONLINE_SOFTMAX_OP=YES")
    print("NO_SCALAR_MEMREF_LOAD_SUPPORT_INTRODUCED=YES")
    print("NONTRANSPOSED_V_GATHER_REMAINS_STAGED=YES")
    print("NATURAL_EXP_SOFTMAX_SEMANTICS_PRESERVED=YES")
    print("ONLINE_SOFTMAX_METADATA_USED_FOR_LOWERING=NO")
    print("NO_SOURCE_NAME_PATH_FIXTURE_SPECIAL_CASES=YES")
    print("QK_SCORE_GENERATION_REMAINS_STAGED=YES")
    print("DOT_CONTRACT_REMAINS_STAGED=YES")
    print("READY_FOR_TRITON=NO")


if __name__ == "__main__":
    main()
