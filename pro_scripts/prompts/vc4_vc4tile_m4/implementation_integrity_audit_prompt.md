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
- `vc4tile.program_id`, `vc4tile.block_id`, or `vc4tile.warp_id` modeled as generic uniform/user-argument reads or carrying `uniform_index`;
- output/input pointers, `n`, `alpha`, strides, or scalar launch arguments represented through `program_id` instead of `vc4tile.kernel` formal arguments;
- `vc4tile.lane_id`, `vc4tile.lane_range`, or `ssavc4.element_number` represented as launch ABI builtins or uniform stream values;
- stale lower ABI builtin categories kept alive after the ABI refactor (`qpu_num`, `num_qpus`, `elem_num`, or `hidden_runtime`).

ABI audit rule: `program_id`, `block_id`, and `warp_id` are runtime builtin identity ops; `lane_id`/`lane_range` are element-number-derived and not uniforms; user/caller values must be kernel formal args. Do not classify a mechanical "fix" to these categories as legitimate without explicit ABI-refactor evidence.

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
