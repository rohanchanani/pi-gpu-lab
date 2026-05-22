# Slice01 Dialect Scaffold

Create the VC4Tile dialect scaffold, CMake/TableGen wiring, vc4-opt registration, and show-dialects smoke. Do not add substantive ops or lowering yet.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.
