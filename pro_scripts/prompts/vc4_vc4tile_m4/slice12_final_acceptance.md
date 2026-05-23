# Slice12 Final Acceptance

Run cumulative M4 contracts, all active VC4Tile hardware fixtures, full check-vc4, M3/M2 regression, and the read-only Codex implementation-integrity audit.

Remember the M4 invariant: `vc4tile` lowers to `ssavc4`, and only then to scheduled `vc4`. Producer adapters are future work after M4.

Final acceptance must include the ABI category audit: user/caller values are `vc4tile.kernel` formal arguments and `vc4.launch_abi.args[]`; `program_id`, `block_id`, and `warp_id` are runtime builtin identity ops and `vc4.launch_abi.builtins[]`; `lane_id`/`lane_range` are `ssavc4.element_number`-derived and not uniforms. The audit must reject mechanical ABI rewrites that hide user arguments behind identity ops or keep stale builtin categories alive.
