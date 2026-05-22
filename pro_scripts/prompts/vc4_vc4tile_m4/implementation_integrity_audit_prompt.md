You are performing a read-only implementation-integrity audit for VC4 Milestone 4.

Inspect the repository freely. Do not write files. Do not stage files. Do not commit.

Authoritative scope:
- M4 defines the VC4 Tile dialect (`vc4tile`) and lowers `vc4tile` to `ssavc4`.
- M4 must not implement Triton, IREE, StableHLO, Torch, JAX, PyTorch, or upstream MLIR `gpu` producer lowering.
- M4 must not lower `vc4tile` directly to scheduled `vc4`.
- The valid lower stack is `vc4tile -> ssavc4 -> scheduled vc4 -> artifacts/runtime`.

Classify whether the implementation appears legitimate. Look specifically for:

- direct `VC4TileToVC4` or `convert-vc4tile-to-vc4` shortcuts;
- producer lowering shortcuts such as `gpu -> ssavc4`, Triton -> `vc4tile`, IREE -> `vc4tile`, or producer -> `vc4`;
- fixture-name or public-name special cases in compiler, emitter, support scripts, or runtime;
- reference QASM/shader substitution;
- fake `VC4_TEST_RESULT` output or synthetic runtime-event injection;
- physical `QPU_NUMBER` used for logical request/block/warp/thread identity;
- weakened M2/M3 artifact, runtime, scheduled-verifier, spill, or hardware contracts;
- expected JSON/reference bundle rewrites used to pass tests;
- support runners that skip the required `vc4tile -> ssavc4 -> scheduled vc4 -> bundle` pipeline;
- barriers accepted without cooperative full-residency requirements;
- shared VPM treated as arbitrary unbounded scalar SRAM;
- arbitrary scatter stores silently lowered through the coalesced VDW path.

Return exactly one JSON object and no markdown:

{
  "integrity_pass": "YES" or "NO",
  "summary": "brief classification",
  "findings": [
    {
      "severity": "blocking" or "warning",
      "path": "repo-relative path if applicable",
      "reason": "what you found"
    }
  ]
}
