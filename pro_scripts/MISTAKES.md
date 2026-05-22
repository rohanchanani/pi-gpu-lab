# VC4 Milestone Automation Mistakes Ledger

This file records concrete automation/specification mistakes that already cost
milestone attempts. Feed it into future milestone package generation and update
it whenever an issue is found, fixed, or generalized.

## 2026-05-22: Generic autorun used a hard-coded M2 Codex prompt during M4

**Symptom:** During M4, Codex mechanical repair prompts still described
"VC4 codegen Milestone 2" and pointed at M2-style files/verification commands
even though the active milestone was VC4Tile M4.

**Root cause:** `vc4_milestone_autorun.py` rendered its Codex prompt from a
hard-coded string instead of the active milestone's prompt package.

**Permanent rule:** Codex mechanical prompts are milestone-specific artifacts.
Every milestone package that allows Codex mechanical repair must provide
`codex_mechanical_prompt.md.j2` and `codex_contract.md` under its
`prompt_template_dir`, and the generic autorun driver must render those files.

**Verification implication:** The milestone package source-product gate should
require the Codex prompt template and contract. If a future milestone introduces
new mechanical repair semantics, update its milestone prompt package rather than
hard-coding new prompt language in the generic driver.

## 2026-05-22: Codex mechanical repair budget was effectively one attempt

**Symptom:** New-dialect build/TableGen/C++ churn consumed GPT attempts while
Codex only had one mechanical repair attempt per category.

**Root cause:** The generic failure router defaulted each mechanical category to
a budget of one, and the M4 worklist set `max_codex_attempts` to one for
implementation slices.

**Permanent rule:** Mechanical Codex repair budget defaults to three attempts
for mechanical categories, especially build/CMake/API/path failures. Milestone
worklists should not set `max_codex_attempts` below three for implementation
slices unless the slice explicitly disables Codex with `codex_policy: "never"`.

**Verification implication:** Future milestone packages should preserve the
three-attempt default and avoid per-slice one-attempt budgets.

## 2026-05-22: `lowered_ir_contract` field mismatch caused `files_checked=[]`

**Symptom:** M4 slice 3 generated an SSAVC4 output file containing the expected
`ssavc4.func`, `ssavc4.thread_end`, `vc4.launch_abi`, and `vc4.resource`
strings, but the verifier failed with `files_checked=[]` and missing literals.

**Root cause:** The M4 verification spec used `files` for
`lowered_ir_contract`, while the verifier implementation inspected
`inspect_files` / `output` / `output_file`. The mechanical smoke covered the
accepted field but not the spec field used by M4.

**Permanent rule:** Text-inspection verification specs should use the canonical
`inspect_files` field. The verifier may support `files` as a compatibility
alias, but new specs should use `inspect_files`.

**Verification implication:** When a verifier failure reports `files_checked=[]`
despite a previous step producing an output file, immediately suspect a spec
field-name mismatch. Fix the verifier/spec pattern across all future slices, not
just the failing gate. Add or update mechanical smoke tests so the field alias
is exercised.

## 2026-05-22: A local workaround marker appeared in slice 3 output

**Symptom:** A failed M4 slice 3 attempt printed a marker like
`// vc4tile-to-ssavc4-lowered: ...` to satisfy a broken text-inspection gate.

**Root cause:** The verifier was inspecting the wrong field, so GPT attempted
to make expected text visible through stdout rather than through the intended
intermediate file.

**Permanent rule:** Do not compensate for verifier/spec bugs by changing
compiler output shape. Fix the verifier/spec, then remove any diagnostic marker
or stdout hack introduced solely to satisfy the broken verifier.

**Verification implication:** After fixing a verifier/spec mismatch, audit the
current worktree and recent failed-attempt patches for workaround markers before
resuming.

## Dialect-contract raw text checks can be gamed by comments

Do not verify dialect operation existence by scanning arbitrary source text for fully qualified operation names such as `vc4tile.program_id`. ODS often stores the operation mnemonic as `"program_id"` and composes the dialect prefix through MLIR, so source code does not naturally need to contain the assembled spelling. Raw scans also let agents satisfy the verifier with comments such as "verifier-required fully-qualified op name".

Persistent rule: `dialect_contract` must distinguish source-native evidence (`op_defs`, `attr_defs`, `type_defs`) from assembled MLIR evidence (`assembly_ops`, `assembly_attrs`). Comments must be stripped before matching, and fully qualified operation names must appear in real MLIR roundtrip/invalid tests or generated roundtrip output, not in verifier-only comments.

## Feature gates must match real feature ownership

If a feature is marked implemented in `feature_gate_contract`, every required verification layer for that feature must exist in the same scope. Do not mark a behavior as its own implemented feature unless the slice also provides the dialect, invalid diagnostic, lowered-IR, scheduled/artifact, and hardware/reference layers that the feature requires. If a behavior is only part of another feature, fold it into that feature or explicitly document the non-required layers.

## Codex must not repair verifier/spec bugs with implementation comments

Codex mechanical repair is allowed to fix compile/build/API/path issues only. If a failure is caused by a verifier/spec mismatch, missing feature layer, or semantic ownership decision, Codex must print `VC4_CODEX_NEEDS_GPT`. It must not add dummy comments, string literals, or source text only to satisfy a verifier scan.

## Stale build-tree lit tests after cancelled or failed attempts

Failed or cancelled attempts can leave copied tests under `compiler/build/test` even after source files are restored. Prefix rechecks can then fail on stale build-tree tests that are not present in the source tree. Failed-attempt cleanup should delete or refresh corresponding `compiler/build/test/...` paths for changed `compiler/test/...` files before cumulative prefix rechecks.
