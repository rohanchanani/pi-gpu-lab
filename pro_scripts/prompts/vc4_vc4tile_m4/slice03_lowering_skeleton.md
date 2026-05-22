# Slice03 Lowering Skeleton

Implement --convert-vc4tile-to-ssavc4 for minimal kernels and metadata. Install run_vc4tile_candidate_codegen_test.sh to preserve input.vc4tile.mlir, lowered.ssavc4.mlir, scheduled.vc4.mlir, and bundle artifacts.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.
