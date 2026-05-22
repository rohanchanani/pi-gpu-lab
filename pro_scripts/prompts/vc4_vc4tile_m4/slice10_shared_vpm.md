# Slice10 Shared Vpm

Implement shared VPM tile alloc/load/store. Treat VPM as row-granular structured 64x16-word tile memory, not arbitrary scalar SRAM. Prove shared_transpose_16x16_vc4tile on hardware.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.
