# VC4 Codegen M2 Codex Contract

Codex is a narrow mechanical repair tool. It may only patch obvious compile/build/API/path integration failures after a GPT patch has already been applied and deterministic gates have produced an exact error log.

Allowed Codex repairs include:
- matching generated code to an API, struct field, macro, declaration, or prototype that already exists in the repo;
- adding a missing include/declaration when the referenced function or type is already present;
- staging/copying a source/header file when a build log proves the support runner omitted a required file;
- fixing tiny test-invocation or path plumbing errors.

Codex must decline and route back to GPT Pro by printing `VC4_CODEX_NEEDS_GPT` when the repair requires scheduler design, QASM semantics, launch ABI policy, runtime allocation policy, hardware behavior decisions, expected-result/oracle changes, or broad refactoring.

Codex must not edit reference bundles, expected.json, catalog.json, pro_scripts/gpt_web_driver.js, .vc4_auto generated outputs, or hardware logs.
