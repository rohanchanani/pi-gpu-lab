# Slice10 Shared Vpm

Implement shared VPM tile alloc/load/store. Treat VPM as row-granular structured 64x16-word tile memory, not arbitrary scalar SRAM. Prove shared_transpose_16x16_vc4tile on hardware.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.

This slice depends on the ABI-8 refactor gate. Runtime metadata such as logical block/warp IDs, VPM base rows, VPM rows, semaphore bases, and resident request IDs belongs in `vc4.launch_abi.builtins[]`; user/caller pointers, sizes, strides, and scalars are `vc4tile.kernel` formal arguments. `lane_id`/`lane_range` remain `ssavc4.element_number`-derived and are not uniform stream values. Codex must not repair ABI category problems mechanically.
