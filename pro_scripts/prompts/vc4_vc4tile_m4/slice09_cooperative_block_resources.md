# Slice09 Cooperative Block Resources

Implement cooperative-block schedule metadata and logical block/warp/thread identity. Prove cooperative_id_writeback_vc4tile on hardware and reject impossible resource requests.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.

Do not start this slice until the ABI-8 refactor gate has passed. `program_id`, `block_id`, and `warp_id` are zero-operand runtime builtin identity ops that lower to `vc4.launch_abi.builtins[]`; user/caller values must be `vc4tile.kernel` formal arguments. `lane_id` and `lane_range` are `ssavc4.element_number`-derived and are not uniforms. Codex must not "fix" any ABI category problem mechanically.
