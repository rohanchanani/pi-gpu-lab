# Slice07 Control Flow Tail Block Args

Implement uniform control flow, tail masks, and simple loops/merges using SSAVC4 successor operands/block args. Reject general divergent side-effecting control flow.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.
