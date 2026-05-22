# Slice02 Full Op Contract

Define the broad VC4Tile op/type/attr contract: kernel, IDs, lane range, masked global memory, rotate/reduce, shared VPM, barrier, resource attrs, traits/effects, roundtrip tests, and invalid tests. Future lowering tests must not be active yet.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.
