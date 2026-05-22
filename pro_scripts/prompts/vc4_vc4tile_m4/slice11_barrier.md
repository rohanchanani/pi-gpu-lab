# Slice11 Barrier

Implement vc4tile.barrier with cooperative full residency, four-semaphore metadata, and uniform participation requirements. Prove qpu_barrier_syncthreads_vc4tile on hardware.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.
