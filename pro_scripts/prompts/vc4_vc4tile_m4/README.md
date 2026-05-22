# VC4 VC4Tile M4 Prompt Package

This prompt package drives Milestone 4: define the VC4 Tile dialect (`vc4tile`) and lower `vc4tile` to the existing `ssavc4` dialect.

The package is intentionally producer-clean. M4 does not implement Triton, IREE, StableHLO, Torch-MLIR, JAX, PyTorch, or MLIR `gpu` producer lowering. Future producer adapters should target `vc4tile` after M4.

The authoritative implementation path is:

```text
vc4tile -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware
```

Every implementation slice must satisfy the canonical M4 feature layers: dialect contract, invalid diagnostic contract, lowered-IR contract, scheduled/artifact contract, and hardware CPU/reference contract when executable.
