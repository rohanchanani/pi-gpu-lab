# Slice11 Barrier

Implement vc4tile.barrier with cooperative full residency, four-semaphore metadata, and uniform participation requirements. Prove qpu_barrier_syncthreads_vc4tile on hardware.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.

This slice depends on the ABI-8 refactor gate. Barrier semaphores and cooperative residency metadata are runtime builtins in `vc4.launch_abi.builtins[]`, not user launch arguments. `program_id`, `block_id`, and `warp_id` remain zero-operand runtime identity ops; `lane_id`/`lane_range` remain `ssavc4.element_number`-derived and outside the uniform stream. Codex must not "fix" ABI semantics mechanically.
