# Slice09 Cooperative Block Resources

Implement cooperative-block schedule metadata and logical block/warp/thread identity. Prove cooperative_id_writeback_vc4tile on hardware and reject impossible resource requests.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.
