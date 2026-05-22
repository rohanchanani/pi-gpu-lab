# Slice05 Masked Global Store

Implement masked coalesced global store through SSAVC4 VDW. Only support coalesced/affine vector stores; reject arbitrary scatter. Prove vector_store_smoke_vc4tile on hardware.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.
