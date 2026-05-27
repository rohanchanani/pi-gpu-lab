# VC4Tile M5 Handoff

M5 starts after M4 and the post-M4/pre-M5 SCF/core-CFG staging work. Expected baseline: VC4Tile core lowering to SSAVC4, `--legalize-vc4tile-core-cfg`, `--verify-vc4tile-core`, substantial SCF hardware fixtures, and green M2/M3/M4 final acceptance.

M5 adds ergonomic surface functionality while keeping VC4Tile core strict. Surface VC4Tile contains tile views, tile loads/stores, copies, reductions, and contracts. Core VC4Tile contains M4-compatible concrete IDs, masks, core arithmetic, core memory, `cf`, and block args.

M5 must not add producer lowering or executable sub-32 precision. Future Triton and IREE/JAX/PyTorch lowerings target the ergonomic VC4Tile surface after M5.
