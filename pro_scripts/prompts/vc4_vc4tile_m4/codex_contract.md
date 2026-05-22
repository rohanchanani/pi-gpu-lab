# VC4 VC4Tile M4 Codex Contract

Codex is a narrow mechanical repair tool. It may only patch obvious compile/build/API/path integration failures after deterministic gates have produced an exact error log.

Allowed Codex repairs include:

- matching generated code to an API, struct field, macro, declaration, or prototype that already exists in the repo;
- adding a missing include/declaration when the referenced function or type is already present;
- staging/copying a source/header file when a support runner clearly omitted a required file;
- fixing tiny test invocation or path plumbing errors.

Codex must decline and route back to GPT Pro by printing `VC4_CODEX_NEEDS_GPT` when the repair requires dialect semantic design, SSAVC4 lowering design, scheduler design, QASM semantics, launch ABI policy, runtime allocation policy, hardware behavior decisions, expected-result/oracle changes, or broad refactoring.

Codex must not edit reference bundles, expected.json, catalog.json, pro_scripts/gpt_web_driver.js, `.vc4_auto` generated outputs, or hardware logs.

Codex must not add verifier-only comments, dummy string literals, or non-semantic source text to satisfy a deterministic verifier scan. If a verifier asks for a fully qualified operation name that source code does not naturally need, Codex must route back to GPT Pro by printing `VC4_CODEX_NEEDS_GPT`; the verifier/spec should be fixed to check ODS definitions and real MLIR assembly tests instead.
