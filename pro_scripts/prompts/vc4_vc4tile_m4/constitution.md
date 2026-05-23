# VC4 VC4Tile M4 Constitution

You are working on Milestone 4 of the Raspberry Pi VideoCore IV MLIR backend.

M4 defines the VC4 Tile dialect (`vc4tile`) and lowers `vc4tile` to the already implemented `ssavc4` dialect. M4 does not lower from Triton, IREE, StableHLO, Torch-MLIR, JAX, PyTorch, upstream MLIR `gpu`, or any other producer IR. Producer adapters into `vc4tile` are future work after M4.

Required stack:

```text
vc4tile
  ↓ --convert-vc4tile-to-ssavc4
ssavc4
  ↓ --convert-ssavc4-to-vc4
scheduled vc4
  ↓ vc4-codegen --emit-bundle
QASM / C / H artifacts / vc4_runtime
```

Hard rules:

- Do not lower `vc4tile` directly to scheduled `vc4`.
- Do not implement `gpu -> ssavc4`, `gpu -> vc4`, Triton -> `vc4tile`, IREE -> `vc4tile`, or any producer adapter in M4.
- Do not resurrect structured `vc4` operations.
- Do not special-case fixture names, public names, expected JSON, QASM, shader arrays, or runtime logs.
- Do not substitute reference QASM or reference shader arrays.
- Do not fake `VC4_TEST_RESULT`.
- Do not use physical `QPU_NUMBER` for logical request, block, warp, lane, or program identity.
- Do not weaken M2/M3 scheduled-verifier, artifact, runtime, spill, or hardware regression contracts.
- Prefer correctness over optimization. Optimization can follow after verified correctness.

ABI value categories are hard rules:

- User/caller values such as output pointers, input pointers, `n`, `alpha`, strides, and scalar parameters must be formal `vc4tile.kernel` arguments. They lower to `vc4.launch_abi.args[]` and generated launch wrapper parameters.
- `vc4tile.program_id`, `vc4tile.block_id`, and `vc4tile.warp_id` are zero-operand runtime builtin identity ops. They lower to `vc4.launch_abi.builtins[]` entries populated by runtime/scheduler metadata. They must never carry `uniform_index` and must never stand in for user launch arguments.
- `vc4tile.lane_id` and `vc4tile.lane_range` are `ssavc4.element_number`-derived lane/register identity values. They are not uniforms and are not `vc4.launch_abi.builtins[]`.
- Physical uniform stream slots are a transport mechanism after lowering, not the semantic category of the VC4Tile value.
- Codex mechanical repair must not "fix" ABI issues mechanically. ABI category, launch ABI schema, runtime packing, and identity semantics require GPT-led design.

`vc4tile` is a tile/kernel IR: target-specific and VC4-aware, but above SSAVC4 machine details. It should expose 16-lane tile intent, masks, global memory intent, cooperative block resources, shared VPM tile intent, and barriers. It should hide TMU token sequencing, VDW setup words, raw semaphore protocol details, physical register allocation, scheduling hazards, branch delay slots, and QASM.
