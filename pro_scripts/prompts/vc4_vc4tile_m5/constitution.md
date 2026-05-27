# VC4 VC4Tile M5 Constitution

You are working on Milestone 5 of the Raspberry Pi VideoCore IV MLIR backend.

M5 is ergonomic VC4Tile only. It extends VC4Tile with surface tile movement, tile/layout metadata, a hardware-aware copy planner, shared VPM tile ergonomics, SCF composition, elementwise tile compute, reductions, and small 32-bit tile contraction/dot/matmul contracts.

Required path:

```text
vc4tile ergonomic surface
  -> --canonicalize-vc4tile-surface
  -> --plan-vc4tile-copies
  -> --legalize-vc4tile-core-cfg
  -> --verify-vc4tile-core
  -> --convert-vc4tile-to-ssavc4
  -> --convert-ssavc4-to-vc4
  -> vc4-codegen --emit-bundle
  -> QASM / C / H artifacts / vc4_runtime
  -> VC4 hardware
```

Hard rules:

- Do not implement Triton, IREE, StableHLO, JAX, PyTorch, Torch-MLIR, upstream MLIR `gpu`, or any other producer lowering in M5.
- Do not create direct `vc4tile -> vc4`, `vc4tile -> QASM`, or producer -> SSAVC4 paths.
- Do not let surface ops, raw `scf.*`, `index` values, or producer dialect ops survive into verified VC4Tile core.
- Do not implement executable f16, bf16, fp8, fp4, int8, uint8, int4, uint4, packed, quantized, or mixed-precision paths in M5.
- Do not fake `VC4_TEST_RESULT`, hard-code fixture results, branch on fixture/public names, substitute host-computed outputs for device outputs, or reuse stale `.vc4_auto` candidates without explicit opt-in.
- Do not weaken M2/M3/M4 regression contracts.

Hardware is the gold standard.
