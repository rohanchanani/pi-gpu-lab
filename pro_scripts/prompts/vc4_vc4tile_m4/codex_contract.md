# VC4 VC4Tile M4 Codex Contract

Codex is a narrow mechanical repair tool. It may only patch obvious compile/build/API/path integration failures after deterministic gates have produced an exact error log.

Allowed Codex repairs include:

- matching generated code to an API, struct field, macro, declaration, or prototype that already exists in the repo;
- adding a missing include/declaration when the referenced function or type is already present;
- staging/copying a source/header file when a support runner clearly omitted a required file;
- fixing tiny test invocation, CMake, lit, PATH, or support-runner plumbing errors;
- cleaning stale candidate/build-test outputs when the source tree has already been restored and the log shows stale lit/candidate reuse.

Codex must decline and route back to GPT Pro by printing `VC4_CODEX_NEEDS_GPT` when the repair requires dialect semantic design, SSAVC4 lowering design, scheduler design, QASM semantics, launch ABI policy, runtime allocation policy, hardware behavior decisions, expected-result/oracle changes, or broad refactoring.

Codex must not edit reference bundles, checked-in `expected.json`, catalog.json, pro_scripts/gpt_web_driver.js, `.vc4_auto` generated outputs, or hardware logs.

Codex must not add dummy comments, string literals, or verifier-only source text to satisfy a scan. If a verifier/spec mismatch causes a failure, Codex must route back to GPT Pro.

Codex must not implement producer lowering from Triton, IREE, StableHLO, Torch, JAX, PyTorch, or upstream MLIR `gpu` in M4, and must not create a direct `vc4tile -> vc4` path.

Codex must not change the meaning of `vc4tile.program_id`, `vc4tile.block_id`, `vc4tile.warp_id`, or `vc4tile.thread_id`. In particular, it must not treat logical identity ops as arbitrary launch-argument/uniform readers. Uniform-argument semantics require a deliberate GPT-led semantic patch.

Codex must preserve the VC4Tile ABI categories:

- user/caller values are `vc4tile.kernel` formal arguments and lower to `vc4.launch_abi.args[]`;
- `vc4tile.program_id`, `vc4tile.block_id`, and `vc4tile.warp_id` are zero-operand runtime builtin identity ops and lower to `vc4.launch_abi.builtins[]`;
- `vc4tile.lane_id` and `vc4tile.lane_range` are `ssavc4.element_number`-derived and are not uniforms or launch ABI builtins.

If a failure asks Codex to add/remove `uniform_index` on VC4Tile identity ops, reinterpret a user value as `program_id`, rename stale launch ABI builtin kinds, change runtime uniform packing, or otherwise "fix" ABI semantics, Codex must leave the worktree unchanged and print `VC4_CODEX_NEEDS_GPT`.

## Milestone-package requirement

This contract is milestone-specific. The generic autorun driver must render Codex mechanical prompts from the active milestone's `prompt_template_dir` (`codex_mechanical_prompt.md.j2` plus this `codex_contract.md`) rather than using hard-coded language from an older milestone. Future milestone packages must provide their own Codex prompt template and contract, and their package-source verification should require both files.
