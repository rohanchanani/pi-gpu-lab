# Slice06 Masked Global Load Saxpy

Implement masked global load through SSAVC4 TMU and prove saxpy_full_vc4tile on hardware. Preserve TMU effect ordering and r4 lifetime through the SSAVC4 lower half.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.
