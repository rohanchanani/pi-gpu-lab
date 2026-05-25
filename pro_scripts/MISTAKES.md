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


## 2026-05-22: Slice-local checks passed before fresh global `check-vc4` failed

**Symptom:** M4 slice 8 passed its own gate and auto-committed, but the cumulative prefix recheck failed back at an earlier slice on `build-check_vc4`.

**Root cause:** The slice ran a global `check-vc4` too early, before later scheduled-artifact and hardware/reference checks exercised fresh candidate/test paths. Stale or divergent generated artifacts could therefore pass the slice-local gate but fail a fresh cumulative check.

**Permanent rule:** Every implementation slice must keep the early `build-check_vc4` gate and also run a final `build-check_vc4-post` after all lowered-IR, scheduled-artifact, hardware/reference, and integrity checks. Fresh global regression must fail inside the owning slice, not only during prefix recheck.

## 2026-05-22: Context profiles used unsupported `kind: glob`

**Symptom:** M4 prompts printed `Unknown include kind glob` for high-value reference tests such as SSAVC4 block-argument, VPM, barrier, and rotate/reduce fixtures.

**Root cause:** The milestone context profiles used `kind: "glob"`, but the generic context packer did not implement glob expansion.

**Permanent rule:** If a milestone context profile uses an include kind, the context packer must support it before the milestone starts. `kind: "glob"` must expand deterministic repo-relative text-file matches, include matched files, and report no-match globs explicitly.

## 2026-05-22: Hardware candidate evidence must be fresh

**Symptom:** A hardware/scheduled-artifact verifier could pass using candidate artifacts under `.vc4_auto` while a fresh `check-vc4` invocation later failed for the same fixture.

**Root cause:** Candidate generation and hardware/reference contracts did not force a clean generation boundary strongly enough.

**Permanent rule:** VC4Tile scheduled-artifact and hardware CPU-reference contracts must clean the relevant candidate bundle before generation and require fresh debug artifacts (`input.vc4tile.mlir`, `lowered.ssavc4.mlir`, `scheduled.vc4.mlir`, manifest/layout, and launch wrappers). Support runners should validate that scheduled output contains exactly one `vc4.module` and no remaining `vc4tile`/`ssavc4` operations before invoking `vc4-codegen`.

## Final check-vc4 must run after slice artifact/hardware checks

A slice can pass an early `check-vc4`, then later scheduled-artifact or hardware
steps can create fresh build-tree/candidate state that only fails during the
cumulative prefix recheck. Every implementation slice should keep the early
`build-check_vc4` gate and also run a final `build-check_vc4-post` after all
slice-owned lowered-IR, scheduled-artifact, hardware/reference, and integrity
checks.


## Context profiles must not use unsupported include kinds

M4 context profiles used `kind: "glob"` before the context packer implemented it,
so prompts reported `Unknown include kind glob` and omitted key SSAVC4 reference
tests. Any new include kind in a milestone context profile must be implemented in
`vc4_codegen_context_pack.py` and should produce deterministic no-match/skip
sections instead of silently losing context.


## Hardware and scheduled-artifact checks must generate fresh candidates

Hardware and scheduled-artifact verifications must not pass by reusing stale
`.vc4_auto` bundles. Scheduled-artifact checks should clean their bundle and
intermediate outputs before generation. Hardware CPU/reference checks should
clean the fixture candidate/hardware directories before running phases. Support
runners should validate that scheduled intermediates contain exactly one
`vc4.module` and no remaining `vc4tile`/`ssavc4` operations before calling
`vc4-codegen`.

## 2026-05-23: program_id was misused as a generic uniform argument

**Symptom:** Some VC4Tile fixtures and lowering paths treated
`vc4tile.program_id` like a generic uniform/user-argument read, including forms
with `uniform_index` or uses where `program_id` stood in for output pointers,
input pointers, `n`, `alpha`, or other caller-supplied values.

**Root cause:** The milestone spec blurred semantic ABI categories with the
physical uniform stream transport. Runtime identity metadata and user kernel
arguments both eventually travel through uniforms, but they are different ABI
categories and have different ownership.

**Permanent rule:** User/caller values are formal `vc4tile.kernel` arguments and
lower to `vc4.launch_abi.args[]`. `vc4tile.program_id`, `vc4tile.block_id`, and
`vc4tile.warp_id` are zero-operand runtime builtin identity ops and lower to
`vc4.launch_abi.builtins[]`. They must never carry `uniform_index` and must not
be used as pointer/scalar launch arguments.

**Verification implication:** Future ABI-refactor verifiers should forbid
`uniform_index` on VC4Tile identity ops, stale lower ABI builtin kinds
(`qpu_num`, `num_qpus`, `elem_num`, `hidden_runtime`), and launch ABI `args[]`
entries named like runtime metadata. These checks must be introduced as an
ABI-refactor gate so already-passed slices do not fail solely because old source
still contains stale lower ABI syntax.

`ssavc4.element_number` stays as the first-class lane/register identity op. It
is not a `vc4.launch_abi.builtins[]` entry and must not be packed into the
uniform stream.

## 2026-05-23: Failed-candidate cleanup discarded useful Codex mechanical repairs

**Symptom:** A Codex mechanical attempt repaired a compile/link/API issue, such
as missing linked `KernelOp::parse` / `KernelOp::print` definitions for custom
ODS parser/printer declarations, but a later semantic gate failed. Cleanup then
discarded the whole failed candidate, so the same mechanical link failure
recurred in the next GPT attempt.

**Root cause:** Failure prompts treated the next GPT attempt as a fresh bundle
request and did not carry forward concise mechanical repairs from the failed
candidate history.

**Permanent rule:** Failure prompts must surface a short "Prior Codex mechanical
repair to preserve" section whenever the packet indicates Codex fixed
compile/API/link/path mechanics. Do not replay old diffs automatically, but do
require the next GPT bundle to preserve the same mechanical constraint.

**Verification implication:** Prompt smoke checks should render a synthetic
Codex failure packet and assert that carry-forward constraints, including
`KernelOp::parse` and `KernelOp::print` when present, appear in the failure
prompt.

## 2026-05-23: GPT failure attempts rewrote already-passing surfaces

**Symptom:** Some failure attempts rewrote broad surfaces that had already
passed, including earlier-slice formal-args, program-id, global-store, SAXPY,
vector-store, warp-reduce, minimal ABI, or cooperative matrix files, instead of
fixing the current failing gate.

**Root cause:** Failure prompts included large context packs but did not state
late and explicitly that a failure attempt is a focused repair attempt, not a
new implementation pass.

**Permanent rule:** Failure prompts must include high-salience repair discipline:
fix the current failure packet, preserve already-passing checks, avoid
unrelated rewrites, and justify any out-of-scope path changes in
`manifest.json risk_notes`.

**Verification implication:** Rendered failure-prompt checks must assert that
focused repair language, `source_product_missing`, `check-vc4`, and
`manifest.json risk_notes` appear in M4 failure prompts.

## 2026-05-23: Resume auto-committed before cumulative prefix validation

**Symptom:** An active slice passed and was auto-committed, but the subsequent
resume-level cumulative prefix recheck failed. HEAD was left at a bad
bad-but-salvageable commit.

**Root cause:** `vc4_milestone_resume.py` delegated to autorun with normal
auto-commit behavior, then ran the cumulative prefix check after the commit had
already landed.

**Permanent rule:** Resume must run autorun slices with `--no-commit`, verify
the active slice and cumulative prefix against the dirty candidate, and only
then perform the same allowlist/forbidden-path guarded commit with the worklist
commit message. If prefix fails, leave the candidate dirty and uncommitted.

**Verification implication:** Static/dry workflow checks should confirm that
resume passes `--no-commit` to autorun and has a deferred guarded commit path
after cumulative prefix success.

<!-- M4_MINIMAL_ABI_PREFIX_CANARY_20260524: mistakes -->

## 2026-05-24: Later M4 slices regressed the m4-03 minimal launch ABI

**Symptom:** Slices m4-09 and m4-10 could pass their active verifier locally but fail cumulative prefix recheck at `m4-03-scheduled-artifact-minimal`. The dirty candidates changed generic VC4Tile launch ABI completion so an empty/minimal kernel acquired a synthetic `total_requests` builtin, `uniform_words_per_qpu = 1`, and an `ssavc4.uniform.read`, which then introduced scheduled hazards in the minimal thread-end pipeline.

**Root cause:** Later-slice implementation attempts repaired local FileCheck/build failures by modifying generic launch ABI normalization and earlier minimal ABI tests instead of preserving the m4-03 prefix invariant.

**Permanent rule:** Empty/no-formal kernels with no used runtime builtins must lower with `args=[]`, `builtins=[]`, `uniform_words_per_qpu=0`, and no `ssavc4.uniform.read`. Runtime builtins belong in `vc4.launch_abi.builtins[]` only when a corresponding vc4tile op/resource uses them. Later slices must not edit m4-03 minimal ABI tests or generic launch ABI completion to satisfy unrelated local failures.

**Verification implication:** m4-10 and later M4 slices must include a minimal ABI prefix canary in their active typed verifier, not only in resume-level cumulative prefix recheck. The canary lowers `minimal-thrend-vc4tile.mlir` through `vc4tile -> ssavc4 -> scheduled vc4 -> vc4-codegen` and asserts the empty ABI invariant above.

## 2026-05-24: Lower-half hardware checks accepted stale or fixed PASS evidence

**Symptom:** Final integrity audit found lower-half hardware-test surfaces where a candidate harness could print `VC4_TEST_RESULT status=PASS` after a failed launch, VC4Tile candidate `run/all` could reuse old generated bundles, and some hardware-labeled reference `run.sh` scripts emitted fixed PASS results without running hardware.

**Permanent rule:** Hardware result lines must be derived from real execution and semantic checks. Candidate runners must generate fresh bundles by default before hardware execution, with any reuse path explicit and opt-in. Fixed PASS reference scripts must be rejected by integrity scans until converted to real hardware-backed reference runs.

## 2026-05-24: Characterization litmus tests were treated as compiler blockers

**Symptom:** `vpm_slice_visibility` was created to characterize VC4 hardware topology and VPM visibility/collision behavior, but milestone acceptance treated it like a required compiler correctness hardware fixture.

**Permanent rule:** Hardware characterization litmus tests such as `vpm_slice_visibility` must stay runnable as manual/post-milestone evidence, but must not be included in required compiler correctness fixture matrices or final acceptance blocker sets. Required compiler matrices should cover generated artifact/runtime correctness, not exploratory topology classification.

## 2026-05-24: M4 final acceptance used historical reference debt as blocker evidence

**Symptom:** M4 final acceptance failed on old lower-half reference bundles that emitted fixed `VC4_TEST_RESULT status=PASS`, even when those reference bundles were not the required M4 hardware evidence path.

**Permanent rule:** M4 final acceptance must use candidate-generated evidence for required hardware fixtures. Historical reference bundles may be audited as technical debt, but they must not become blockers unless a required verification uses them as evidence. Required hardware proof means generated artifacts plus candidate harness plus device copyback plus host oracle comparison.

**Verification implication:** Required M4 hardware verifications must force fresh candidate generation by default. Stale generated candidate reuse is not valid acceptance evidence unless an explicit opt-in environment variable such as `VC4_REUSE_GENERATED_CANDIDATE=1` is used for local debugging outside the required acceptance path.
