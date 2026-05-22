# Slice04 Independent Tile Ids Arith Masks

Implement program/request IDs, lane range, arithmetic, comparisons, and masks. Lower lane ID through ssavc4.element_number and logical IDs through launch ABI uniforms. Never use QPU_NUMBER for logical identity.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.
