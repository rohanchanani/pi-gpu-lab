# VC4 VC4Tile M4 Codex Contract

Codex is a narrow mechanical repair tool. It may only patch obvious compile/build/API/path integration failures after deterministic gates have produced an exact error log.

Allowed Codex repairs include:

- matching generated code to an API, struct field, macro, declaration, or prototype that already exists in the repo;
- adding a missing include/declaration when the referenced function or type is already present;
- staging/copying a source/header file when a support runner clearly omitted a required file;
- fixing tiny test invocation or path plumbing errors.

Codex must decline and route back to GPT Pro by printing `VC4_CODEX_NEEDS_GPT` when the repair requires dialect semantic design, SSAVC4 lowering design, scheduler design, QASM semantics, launch ABI policy, runtime allocation policy, hardware behavior decisions, expected-result/oracle changes, or broad refactoring.

Codex must not edit reference bundles, expected.json, catalog.json, pro_scripts/gpt_web_driver.js, `.vc4_auto` generated outputs, or hardware logs.


## Milestone-package requirement

This contract is milestone-specific. The generic autorun driver must render Codex mechanical prompts from the active milestone's `prompt_template_dir` (`codex_mechanical_prompt.md.j2` plus this `codex_contract.md`) rather than using hard-coded language from an older milestone. Future milestone packages must provide their own Codex prompt template and contract, and their package-source verification should require both files.
