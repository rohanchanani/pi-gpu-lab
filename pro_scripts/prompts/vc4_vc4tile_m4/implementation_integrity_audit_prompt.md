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
- fake `VC4_TEST_RESULT` output or synthetic runtime-event injection in candidate harnesses, support runners, generated artifacts, or required verification paths;
- stale generated candidate reuse as acceptance proof unless reuse is explicit opt-in for a non-acceptance debugging run;
- reference QASM/shader substitution only when a required verification uses reference QASM, reference shader arrays, or `reference/run.sh` as acceptance evidence;
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

Classify these as blocking:
- direct `vc4tile -> vc4` lowering;
- producer-to-SSAVC4 or producer-to-VC4 shortcuts;
- fixture-name special casing in lowering, emission, support runners, or runtime;
- `expected.json` edits that hide runtime failure or tolerate failed hardware;
- fake `VC4_TEST_RESULT` in required VC4Tile candidate harnesses or the VC4Tile candidate support runner;
- stale generated candidate reuse as default acceptance evidence;
- use of physical `QPU_NUMBER` for logical program/block/warp/lane identity;
- weakening M2/M3 regression coverage, especially replacing M3 `--slice all` with only M3 final acceptance;
- use of lower-half `reference/run.sh` as required M4 final-acceptance evidence.

Classify these as non-blocking by themselves:
- historical lower-half reference directories under `compiler/test/CodeGen/VC4/Hardware/Run/**/reference/**` when no required M4 final-acceptance verification uses them;
- old fixed-PASS lower-half reference scripts when not used as required M4 acceptance evidence;
- `vpm_slice_visibility` characterization artifacts when not in M4 required fixture matrices and not used as required M4 final-acceptance evidence.

ABI audit rule: `program_id`, `block_id`, and `warp_id` are runtime builtin identity categories; `lane_id`/`lane_range` are element-number-derived and not uniforms; user/caller values must be kernel formal args. For this M4 final-acceptance cleanup, do not require a new ABI refactor or reject the existing ABI-1 implementation solely because logical identity categories are materialized through launch ABI builtins or `ssavc4.uniform.read`. Treat that representation as non-blocking when the values remain logical runtime identities, are not physical `QPU_NUMBER`, and are not reused as user pointers, user buffers, or scalar kernel arguments. Continue to block category abuse, physical-QPU identity, user/caller values modeled as `program_id`, and expected/runtime edits that hide ABI failures.

Classification rule: `compiler/test/CodeGen/VC4/Hardware/Run/vpm_slice_visibility` is a VC4 hardware topology/VPM visibility characterization litmus, not a required compiler correctness hardware fixture. Do not make M4 integrity fail solely because this fixture is not a normal generated compiler-correctness proof. It may still be mentioned as non-blocking characterization context. Continue to treat fake `VC4_TEST_RESULT status=PASS`, reference substitutions, or stale-candidate reuse in required compiler correctness fixtures as blocking.

M4 final acceptance evidence rule: required hardware proof is candidate-first:
`input.mlir -> vc4-opt lowering -> vc4-codegen emitted artifacts -> candidate host harness -> Pi hardware run -> device output copied back -> host oracle comparison -> VC4_TEST_RESULT reflecting mismatches and launch failures`. Data-producing required fixtures must have device output copied back and compared against a host oracle. A pure synchronization/liveness fixture with no user output, no global store, and an `expected.json` contract limited to launch/runtime completion fields may use runtime launch success/failure counters as its device-derived result; do not classify that narrow no-output barrier shape as fake solely because there is no output buffer to copy back. Do not fail solely because historical lower-half reference directories exist. Treat historical reference bundles as blocking only when a required M4 verification uses their QASM, shader arrays, `expected.json` rewrites, runtime logs, or `reference/run.sh` as acceptance evidence.

Base the M4 acceptance-evidence decision on `pro_scripts/vc4_vc4tile_m4_verifications.json` and `defaults.fixture_matrices.m4_required`, not on arbitrary files merely existing elsewhere in the tree.

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
