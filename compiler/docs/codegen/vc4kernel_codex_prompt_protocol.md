# VC4Kernel Codex Prompt Protocol

**Recommended repo path:** `compiler/docs/codegen/vc4kernel_codex_prompt_protocol.md`  
**How to use:** prepend future implementation prompts with:

```text
Read compiler/docs/codegen/vc4kernel_codex_prompt_protocol.md before starting. Follow it as a standing protocol for this prompt.
```

This document is a standing behavior contract for Codex runs in `rohanchanani/pi-gpu-lab` on branch `compiler`, especially during the `vc4kernel` strict-surface hardening sequence.

The specific task prompt still defines the concrete implementation work. This document defines how to interpret success, failure, commits, verification, scope boundaries, and anti-shortcut rules consistently.

---

## 1. Normative project contract

Treat `compiler/docs/codegen/vc4kernel_dialect_strict_specification.md` as the acceptance contract for the current `vc4kernel` stage.

There is no compatibility or alias phase for accepted `vc4kernel` IR. Anything not explicitly permitted by the strict specification is forbidden in verified `vc4kernel` IR.

The intended stack is:

```text
Triton TTIR / tt dialect
  -> standard MLIR value layer: vector + arith + memref + scf/cf + minimal VC4 plumbing
  -> vc4kernel
  -> ssavc4
  -> scheduled/register-allocated vc4
  -> qasm/c/h emitted bundle
  -> libpi-backed real VC4 hardware
```

Current strict-surface work is surface-only unless the task prompt explicitly says otherwise. Surface-only means:

```text
- dialect identity
- op/type/attr inventory
- parsing/printing/roundtrip behavior
- verifier behavior
- deterministic invalid diagnostics
- static integrity checks
```

Surface-only does **not** mean:

```text
- completing vc4kernel -> ssavc4 lowering
- adding new hardware fixture matrices
- implementing vector -> vc4kernel
- implementing Triton -> vector
- deleting VC4Tile before the surface lock is accepted
```

---

## 2. Non-negotiable scope boundaries

Do not do any of the following unless the current prompt explicitly requests it:

```text
- do not complete or broaden vc4kernel -> ssavc4 lowering
- do not add hardware fixtures during surface-only prompts
- do not introduce vector/memref/Triton lowering
- do not introduce TTIR, TTGIR, gpu, IREE, StableHLO, Torch, or producer-integration work
- do not delete VC4Tile during surface-lock prompts
- do not weaken SSAVC4 or scheduled-VC4 verifiers
- do not add a direct VC4KernelToVC4 path
- do not route around ssavc4
- do not make the verifier accept forbidden dialects, forbidden types, or old vc4tile surface ops
```

Forbidden shortcuts include:

```text
- fixture-name special cases
- public_name special cases
- comments or dummy strings added only to satisfy scans
- reference QASM substitution
- expected-output substitution
- host-side computation replacing device computation
- fixed VC4_TEST_RESULT PASS strings
- stale candidate reuse in an acceptance gate
- weakening diagnostics to make tests pass
- deleting meaningful failing tests instead of fixing the implementation
- direct vc4kernel -> scheduled vc4 lowering
```

If passing a prompt would require any shortcut above, report `FAILURE`.

---

## 3. Worktree and commit protocol

Every run must begin with:

```bash
git status --short --branch
```

### 3.1 Commit previous successful prompt only when moving on

If the previous prompt ended with `SUCCESS` and the user now sends a new prompt that clearly indicates moving on to the next prompt/slice, commit the previous prompt's work **before** starting the new task.

Clear move-on signals include messages like:

```text
Proceed with the next prompt.
Carry on.
Run B2.
Here is the next prompt.
Let's move on.
Continue the sequence.
```

Do **not** commit merely because you reported `SUCCESS`. Wait for a later user prompt that clearly indicates the previous success landed and the user wants to proceed.

If the current task prompt says `Do not commit`, interpret that as:

```text
Do not commit the current prompt's new work after reporting SUCCESS.
```

Do not interpret it as a prohibition on committing the previous successful prompt at the start of a clearly next/move-on prompt. The previous-work commit is a separate confirmation step triggered by the user's move-on prompt.

### 3.2 Conditions required before committing previous work

Before committing previous work, confirm all of the following:

```text
- the previous assistant/Codex final response started with SUCCESS
- the dirty file set matches the modified-file list reported in that previous SUCCESS response
- no unrelated dirty user changes are present
- no generated test artifacts or transient files are being committed accidentally
- git diff --check passes
```

Use explicit file paths with `git add`. Do not use `git add .` unless the prompt explicitly permits it and the status is trivially safe.

Suggested previous-work commit flow:

```bash
set -euo pipefail
git status --short --branch
git diff --check
git diff --name-only
# Inspect the dirty paths and confirm they are exactly the previous SUCCESS paths.
git add <exact previous-success files>
git commit -m "<concise message for previous prompt>"
git status --short --branch
```

If the dirty set does not match the previous success, or if you cannot confidently distinguish previous work from unrelated user changes, report `FAILURE` before editing.

### 3.3 End-of-current-prompt commit rule

At the end of a successful current prompt, do **not** commit unless the current prompt explicitly asks you to commit.

Instead, report:

```text
- exact modified files
- exact verification commands run
- key source/test evidence
- final git status
```

The next move-on prompt is the confirmation point for committing this work.

---

## 4. SUCCESS / FAILURE response contract

The final response must start with exactly one of these words on the first line:

```text
SUCCESS
```

or:

```text
FAILURE
```

No Markdown heading or prefix may appear before that word.

### 4.1 Use SUCCESS only when all required verification passes

Report `SUCCESS` only if all of the following are true:

```text
- the requested implementation changes were made honestly within scope
- every required verification command from the prompt was run
- every required verification command passed in the final state
- every source/static check required by the prompt passed
- git diff --check passed
- git status --short shows only intentional files for this prompt
- no forbidden shortcut was used
```

### 4.2 FAILURE means genuinely blocked, not first failed command

Do not report `FAILURE` just because the first attempted command failed.

If a verification command fails for a reason you can fix honestly within the prompt's scope, fix it and rerun the required verification. This includes narrow, deterministic fixes such as:

```text
- fixing syntax in a newly added invalid test
- fixing an expected diagnostic string to match the intended deterministic diagnostic
- adding a focused lit.cfg.py for a test directory when the source-path lit invocation is otherwise non-deterministic
- fixing parser/ODS/verifier code to enforce the stricter requested contract
- removing generated test detritus produced by your own command when it is clearly transient and should not be committed
```

Use `FAILURE` only if you are genuinely blocked, such as:

```text
- the prompt or strict spec is ambiguous in a way that requires a human design decision
- the current repo state has unrelated dirty user changes you cannot preserve confidently
- the requested behavior is incompatible with the current architecture
- passing would require weakening the verifier or diagnostics
- passing would require a fixture-name/public_name shortcut
- passing would require host-side substitution, reference-output substitution, or direct VC4KernelToVC4 lowering
- passing would require broad unrelated infrastructure changes outside the prompt's scope
- required verification cannot be made deterministic without a non-local design/infrastructure change
```

If you report `FAILURE`, include:

```text
- the precise blocker
- the exact failing command
- the relevant output snippet
- why honest mechanical iteration is not viable
- current modified files, if any
```

---

## 5. Verification discipline

Run every required command in the task prompt. Do not omit a command because a stronger-looking command passed.

If the prompt provides a focused lit helper, use it exactly. If no helper is provided, prefer:

```bash
run_lit() {
  if command -v llvm-lit >/dev/null 2>&1; then
    llvm-lit "$@"
  elif [ -x /opt/homebrew/opt/llvm/bin/llvm-lit ]; then
    /opt/homebrew/opt/llvm/bin/llvm-lit "$@"
  else
    python3 -m lit "$@"
  fi
}
```

For `vc4kernel` surface-hardening prompts, the usual final verification block should include at least:

```bash
set -euo pipefail
ninja -C compiler/build vc4-opt
run_lit -sv compiler/test/Dialect/VC4Kernel
ninja -C compiler/build check-vc4
git diff --check
git status --short
```

If the current prompt specifies additional checks, run those too.

Passing `ninja -C compiler/build check-vc4` does not replace a required focused source-path lit command unless the prompt explicitly says so. If the source-path lit command fails due to a local deterministic lit configuration problem, fix it narrowly if possible and rerun.

---

## 6. Lit/test infrastructure policy

A narrow test-harness fix is allowed only when it makes required verification deterministic without weakening semantics.

Allowed examples:

```text
- adding a focused lit.cfg.py under compiler/test/Dialect/VC4Kernel so source-path lit can find vc4-opt, FileCheck, and not
- changing a focused lit exec root so .lit_test_times.txt is not written into the source tree
- correcting a newly added invalid test so the intended verifier diagnostic is reached before an unrelated verifier failure
```

Not allowed:

```text
- globally rewriting lit infrastructure during a surface verifier prompt
- disabling failing tests
- changing RUN lines to avoid the relevant verifier
- making invalid tests fail only by parse errors when the prompt requires semantic diagnostics
- hiding failing tests behind unsupported/xfail unless the prompt explicitly asks
```

If a generated transient file appears from a test run, remove it only when it is clearly generated by the run and not a user file. If unsure, report `FAILURE`.

---

## 7. Source and test evidence expectations

For verifier/surface prompts, changes should normally include both implementation and tests.

Implementation evidence should be in real source files such as:

```text
compiler/include/vc4/Dialect/VC4Kernel/IR/...
compiler/lib/Dialect/VC4Kernel/IR/...
compiler/lib/Conversion/VC4KernelToSSAVC4/...
```

Test evidence should normally include deterministic lit tests under:

```text
compiler/test/Dialect/VC4Kernel/
compiler/test/Conversion/VC4KernelToSSAVC4/
```

Invalid tests should check deterministic semantic diagnostics with `FileCheck`, not just parser failures, unless the prompt explicitly says a parser rejection is the intended boundary.

Static/source checks should inspect real implementation constructs. Do not satisfy scans by adding unrelated comments or string literals.

---

## 8. VC4Kernel strict-surface expectations

During the strict-surface hardening sequence, preserve these expectations:

```text
- final op/type/attr inventory follows vc4kernel_dialect_strict_specification.md
- no vc4kernel.thread_id
- no vc4kernel.tile_* ergonomic surface ops
- no vector dialect operations inside verified vc4kernel
- no memref/tensor/scf/func/linalg/gpu/tt/ttg/nvgpu/nvvm/rocdl/spirv/iree/stablehlo/mhlo/llvm inside verified vc4kernel
- no ssavc4 or scheduled vc4 ops inside verified vc4kernel
- vector<16xi32> and vector<16xf32> carrier types are allowed, but vector.* operations are not
- vector<16xi1>, sub-32 types, index, memref, tensor, pointers, and unknown vc4kernel types are rejected
- arith is allowed only in the scalar subset specified by the strict spec
- cf is allowed only as specified by the strict spec
- predicates must be normalizable where consumers require that
- memory path ops must enforce alignment/contiguity/resource constraints conservatively
- VPM ops must be statically resource-accounted and bounds-checked
```

If a prompt conflicts with these expectations in a way that weakens the strict spec, report `FAILURE` and explain the conflict.

---

## 9. Final response format

A successful final response should look like:

```text
SUCCESS

Implemented <prompt ID / short description>.

Modified files:
- <path>
- <path>

Key evidence:
- <specific source change>
- <specific test/diagnostic added>
- <specific static check result>

Verification run:
- <exact command>: passed, <important count/output if relevant>
- <exact command>: passed
- git diff --check: passed
- git status --short: only intentional files listed below

Final status:
<git status --short output>
```

A failure final response should look like:

```text
FAILURE

Blocked on <precise issue>.

Why this is blocked:
- <why honest iteration is not viable>

Failing command:
<command>

Output snippet:
<snippet>

Modified files, if any:
- <path>
```

---

## 10. One-paragraph rule

Iterate honestly until the requested strict implementation and all required verification pass. Report `SUCCESS` only after the final verified state is clean and in scope. Report `FAILURE` only when you are genuinely blocked by ambiguity, unrelated repo state, architecture incompatibility, non-deterministic verification that cannot be fixed narrowly, or a requirement that would force a shortcut. Commit previous successful work only at the start of the next clearly move-on prompt, never merely because you just reported success.
