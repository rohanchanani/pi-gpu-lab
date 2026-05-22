# Slice08 Rotate Reduce

Implement warp-local rotate/reduce lowering through SSAVC4 rotate/reduction support and prove warp_reduce_sum_vc4tile on hardware.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.
