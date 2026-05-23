# VC4 VC4Tile M4 Slice Contract

Each slice must preserve the milestone path:

```text
vc4tile input
  -> vc4-opt --convert-vc4tile-to-ssavc4
  -> ssavc4
  -> vc4-opt --convert-ssavc4-to-vc4
  -> scheduled vc4
  -> existing scheduled verifiers
  -> vc4-codegen --emit-bundle
  -> existing artifact/runtime path
```

Feature implementation rule:

```text
Every implemented vc4tile feature must pass:
  dialect contract
  invalid diagnostic contract
  vc4tile -> ssavc4 lowered-IR contract
  ssavc4 -> vc4 scheduled/artifact contract
  hardware CPU-reference contract when executable on hardware
```

Hardware is the gold standard. If the feature can be executed on VC4 hardware, it should have a hardware fixture and reference comparison. If a feature is metadata-only or intentionally not yet executable, the feature gate must document why hardware is not required for that feature in that slice.

No slice may make future-slice tests active before the owning implementation exists. Global `check-vc4` must remain green.

Implementation slices must run `check-vc4` both before and after slice-owned scheduled/artifact/hardware verifications. The final post-artifact `check-vc4` catches stale candidate reuse, build-tree lit drift, and fresh-regeneration failures that an early build check cannot see.

Candidate runners must treat generated intermediates as disposable. A `generate` phase must clean stale candidate outputs before regenerating `input.vc4tile.mlir`, lowered SSAVC4, scheduled VC4, manifest/layout, launch wrappers, QASM, and shader arrays. Stale `.vc4_auto` bundles are never proof of a feature.

Scheduled-artifact checks must validate the fresh scheduled VC4 intermediate before invoking `vc4-codegen`: exactly one top-level `vc4.module`, no remaining `vc4tile` or `ssavc4` operations, and at least one scheduled `vc4.qpu.*` operation for executable kernels.

Hardware matrix parameters must be real inputs to the runner/harness, not decorative JSON. The verifier exports matrix values as `VC4_MATRIX_*` / `VC4_CASE_*`; runners and harnesses must consume them or the feature should not claim matrix coverage.

ABI category contract:

- User/caller values must be `vc4tile.kernel` formal arguments and lower to `vc4.launch_abi.args[]`.
- `program_id`, `block_id`, and `warp_id` are runtime builtin identity ops that lower to `vc4.launch_abi.builtins[]`; they are not generic uniform readers and must not carry `uniform_index`.
- `lane_id` and `lane_range` are derived from `ssavc4.element_number`; lane identity is not a uniform slot and must not appear in `vc4.launch_abi.builtins[]`.
- Codex must not repair ABI category failures mechanically. If a gate exposes an ABI semantic issue, route it back to GPT Pro.
