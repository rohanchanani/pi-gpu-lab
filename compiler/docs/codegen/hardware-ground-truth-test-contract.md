# VC4 CodeGen Hardware Ground-Truth Test Contract

Status: **locked contract for VC4 codegen ground-truth tests**  
Audience: VC4 backend maintainers and coding agents authoring tests  
Applies to: `compiler/test/CodeGen/VC4/**` and `compiler/docs/codegen/test-backlog.md`

This document is the source of truth for the VC4 codegen hardware-test corpus. It supersedes earlier “hardware-only reference” drafts and incorporates the first successful `minimal_thrend` hardware run.

The locked test model is:

1. Each hardware-run test has an **input MLIR program** in final-stage `vc4` dialect form.
2. Each hardware-run test has a trusted **reference bundle** that computes the same semantics on real Raspberry Pi VC4 hardware.
3. Before codegen exists, the test validates only the **reference / ground-truth side**.
4. After codegen exists, the same test also generates a **candidate bundle** from `input.mlir`, builds it, runs it on hardware, and checks the candidate result against the same semantic oracle.

The goal is not to make generated qasm text character-for-character identical to the reference qasm. The goal is to prove that codegen from the given `vc4` MLIR produces a deployable bundle that computes the same result on hardware.

---

## 1. Hardware and project basis

The VC4 backend targets QPU user programs, not merely textual assembly output. Runnable examples must therefore be grounded in actual QPU launch and execution behavior.

Relevant VC4 hardware facts that shape this contract:

- QPUs are 16-way SIMD processors.
- General-purpose user programs are queued through the V3D QPU scheduler user-program request interface. The host provides a program address and uniforms address / length to the scheduler, and observes request/completion state through V3D scheduler registers.
- Uniforms are a sequential stream. Reading the uniform register consumes the next 32-bit word and auto-increments the uniforms pointer.
- Program termination uses a thread-end / program-end signal and two following delay-slot instructions.
- TMU, SFU, VPM/VDR/VDW, semaphore, mutex, branch, and thread-switch behavior have real placement and hazard constraints that must ultimately be validated on hardware or explicitly categorized as assembler-only / litmus-only until hardware execution exists.

The practical conclusion is:

> For runnable examples, **real Raspberry Pi hardware execution is the gold standard**. Local checks verify shape, syntax, and buildability. Semantic correctness is established by the hardware result line.

---

## 2. Test phases

Every ordinary codegen hardware-run test has up to two runnable sides.

### 2.1 Reference / ground-truth side

This side exists first.

It contains a trusted, self-contained qasm + C launcher + C harness bundle that can run on the Pi without codegen. The bundle may be hand-authored, imported from a known-good reference, or otherwise explicitly trusted by maintainers.

The reference side proves:

- the semantic behavior expected from the test,
- the runtime/build/harness assumptions,
- the launcher ABI shape for the example,
- the relevant hardware behavior exercised by the kernel.

The reference side is what we create now.

### 2.2 Candidate / generated-code side

This side exists later, after codegen exists.

The candidate side is produced from `input.mlir` by the VC4 code generator. It must produce a source bundle:

- `kernel.qasm`,
- `kernel_launch.c`,
- `kernel_launch.h`,

plus whatever local test harness glue is needed to run it through the same Pi flow.

The candidate side passes only if its semantic result matches the same expected JSON oracle used by the reference side.

---

## 3. Implemented hardware-run directory shape

Each implemented hardware-run test lives under:

```text
compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/
```

Required shape during the reference-only phase:

```text
compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/
  README.md
  input.mlir
  expected.json

  reference/
    .gitignore
    Makefile
    run.sh
    3-test-<test-name>.c
    mailbox.c
    mailbox.h
    <kernel>.qasm
    <kernel>_launch.c
    <kernel>_launch.h

  share/                         # required when vc4asm -c needs ../share templates/includes
    vc4tmpl/template.h           # copied mechanically from old_compiler/reference/share
    vc4inc/vc4.qinc              # required if qasm includes ../share/vc4inc/vc4.qinc
```

Optional files:

```text
  input.json                     # deterministic host-side input description, if useful
  reference_result.golden.json   # optional captured reference result if stable
  notes.md                       # extra hardware notes, if needed

  candidate/
    README.md                    # optional placeholder before codegen exists
```

There is **no required root-level `run.sh`** in the locked contract. The support runner is side-aware and runs `reference/run.sh` or `candidate/run.sh` directly.

Generated/transient files must not be committed unless a test README explicitly justifies them as source-of-truth artifacts. Normally, do **not** commit:

```text
objs/
*.o
*.d
*.elf
*.bin
*.list
*shader.c
*shader.h
run.log
candidate/generated/*
```

The reference directory must normally contain a `.gitignore` like:

```gitignore
objs/
*.o
*.d
*.elf
*.bin
*.list
*shader.c
*shader.h
run.log
```

---

## 4. `input.mlir` contract

`input.mlir` is the MLIR program that the future code generator should consume. It must describe the same kernel/launch semantics as the reference bundle.

Because `compiler/test/lit.cfg.py` discovers `*.mlir`, every checked-in `input.mlir` under `compiler/test` must also be a valid lit test. A hardware-run `input.mlir` is therefore both:

1. the future codegen input, and
2. a local lit-checked final-stage `vc4` emission-contract input.

### 4.1 Required `RUN:` line

Every hardware-run `input.mlir` must begin with a `RUN:` line that verifies the final-stage VC4 emission contract locally.

Recommended v1 scheduled-kernel `RUN:` line:

```mlir
// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null
```

This prevents lit from reporting the file as unresolved and gives the test a useful local shape check even before candidate-side codegen exists.

### 4.2 Required MLIR properties

`input.mlir` must:

1. use the `vc4` dialect,
2. represent the final stage before code generation,
3. be parseable by current `vc4-opt`, unless the test is explicitly not yet cataloged,
4. identify the launchable QPU kernel,
5. carry the launcher ABI metadata needed to generate `launcher.c` and `launcher.h`,
6. make the physical uniform stream layout explicit through verified metadata,
7. make the tail policy explicit,
8. match the semantic behavior of the reference bundle.

### 4.3 Final-stage device body

For v1 codegen, qasm emission consumes only scheduled QPU sink functions:

```mlir
vc4.func @kernel_name() attributes {
  domain = #vc4.execution_domain<qpu>,
  form = #vc4.function_form<scheduled>,
  kernel,
  threading = #vc4.threading_mode<single>,
  "vc4.launch_abi" = { ... }
} {
  // vc4.qpu.bundle / vc4.qpu.ldi / vc4.qpu.sema / vc4.qpu.branch only.
}
```

The scheduled sink subset for final qasm emission is:

- `vc4.qpu.bundle`,
- `vc4.qpu.ldi`,
- `vc4.qpu.sema`,
- `vc4.qpu.branch`.

Structured `vc4` functions are not final qasm-emission inputs. They are lowering inputs for later milestones.

---

## 5. Launcher ABI metadata contract

The qasm body alone is not enough to generate the launcher. The launcher also needs semantic argument and physical uniform ABI metadata.

The current dialect supports this with a verified `"vc4.launch_abi"` dictionary attribute on launchable QPU kernel functions. This attribute is part of the hardware-run input contract.

### 5.1 Required top-level fields

A kernel function with `"vc4.launch_abi"` must be a `kernel` function in QPU domain.

The dictionary must contain:

```mlir
"vc4.launch_abi" = {
  public_name = "...",
  tail_policy = "exact_multiple" | "tail_safe",
  uniform_words_per_qpu = <positive i32>,
  args = [...],
  builtins = [...]
}
```

Meaning:

- `public_name`: public launcher function name in generated C.
- `tail_policy`: test/kernel policy for non-full vector tails.
- `uniform_words_per_qpu`: number of 32-bit words in the per-QPU physical uniform stream.
- `args`: semantic public API arguments.
- `builtins`: execution builtins materialized for the kernel.

Uniform indices across `args` and uniform-materialized `builtins` must be unique and dense in `[0, uniform_words_per_qpu)`.

### 5.2 Argument entries

Buffer argument:

```mlir
{name = "x", kind = "buffer", direction = "in" | "out" | "inout", elem_type = "i8" | "u8" | "i16" | "u16" | "i32" | "u32" | "f32", uniform_index = 0 : i32}
```

Scalar argument:

```mlir
{name = "n", kind = "scalar", direction = "by_value", type = "i32" | "u32" | "f32" | "index", uniform_index = 1 : i32}
```

The public launcher API exposes semantic arguments. It must not expose raw uniform arrays, `qpu_id`, `num_qpus`, hardware addresses, or scheduler registers unless a specific low-level test explicitly exists for such an API.

### 5.3 Builtin entries

Uniform-suffix builtin:

```mlir
{name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 4 : i32}
{name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 5 : i32}
```

Register-materialized builtin, if used:

```mlir
{name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "register"}
```

Rules:

- `#vc4.builtin_kind<elem_num>` must not appear in `vc4.launch_abi` builtins.
- `#vc4.builtin_kind<num_qpus>` currently uses `materialization = "uniform_suffix"`.
- register-materialized builtins must not specify `uniform_index`.

### 5.4 SAXPY ABI example

For the SAXPY reference, the semantic public API is:

```c
int saxpy_launch(struct vc4_runtime *rt, float *x, float *y, float a, uint32_t n);
```

The physical uniform stream per QPU is:

```text
[0] x base address
[1] y base address
[2] alpha as one f32/u32 word
[3] n element count
[4] qpu_id
[5] num_qpus
```

The last two entries are conceptual execution builtins even if the current physical implementation passes them through the uniform stream.

---

## 6. Reference bundle contract

The reference bundle is a trusted implementation of the same semantics as `input.mlir`.

Required files in `reference/`:

```text
reference/
  .gitignore
  Makefile
  run.sh
  3-test-<test-name>.c
  mailbox.c
  mailbox.h
  <kernel>.qasm
  <kernel>_launch.c
  <kernel>_launch.h
```

For examples modeled on the known SAXPY reference, `reference/run.sh` should:

1. assemble qasm with `vc4asm`,
2. generate derived shader C/H files,
3. build the bare-metal test using the local Makefile,
4. boot/run the Pi with `pi-install` or the configured equivalent,
5. stream serial output to stdout.

The reference `run.sh` must not power-cycle the Pi. Power cycling is the job of `Support/run_hardware_test.sh`.

### 6.1 vc4asm `share/` requirement

When `reference/run.sh` runs `vc4asm -c ...` from inside `reference/`, vc4asm may look for generated-C templates at:

```text
../share/vc4tmpl/template.h
```

Therefore tests using `vc4asm -c` must either:

1. copy `old_compiler/reference/share` to the test root as `<test-root>/share`, or
2. document another explicit template/include strategy.

For qasm that includes vc4asm helper macros, prefer the stable relative include form used by the reference tests:

```qasm
.include "../share/vc4inc/vc4.qinc"
```

Do not rely on a machine-local absolute include path inside checked-in qasm.

---

## 7. `expected.json` contract

Every hardware-run test root contains:

```text
expected.json
```

The expected JSON is the semantic oracle. It is intentionally small and stable.

Example:

```json
{
  "name": "minimal_thrend",
  "status": "PASS",
  "required": {
    "completed_qpus": 12
  }
}
```

Example with float tolerance:

```json
{
  "name": "saxpy_reference",
  "status": "PASS",
  "required": {
    "mismatches": 0
  },
  "float_max": {
    "max_abs_diff": 0.0001
  }
}
```

Meaning:

- top-level `name` is required exactly,
- top-level `status` is required exactly,
- fields under `required` are exact comparisons after type coercion,
- fields under `float_max` require `abs(actual) <= limit`.

Do not include unstable values such as execution time, speedup, serial port names, or build paths in required checks.

---

## 8. `VC4_TEST_RESULT` contract

Every successful hardware run must print a final machine-readable line:

```text
VC4_TEST_RESULT name=<test-name> status=PASS key=value ...
```

Rules:

1. The checker uses the **last** `VC4_TEST_RESULT` line in the log.
2. Every field is `key=value`.
3. Values should avoid whitespace; use underscores if needed.
4. The line must include at least:
   - `name=<test-name>`
   - `status=PASS`
5. Tests should include semantic fields such as:
   - `mismatches=0`
   - `max_abs_diff=0.0`
   - `completed_qpus=12`
   - `checksum=<value>`
6. Timing values may be printed but must not be required by `expected.json` unless the test is explicitly about timing.

Example from the first successful hardware contract test:

```text
VC4_TEST_RESULT name=minimal_thrend status=PASS completed_qpus=12 active_qpus=12 elapsed_usec=100
```

The oracle for that test requires only `completed_qpus=12`, not `elapsed_usec`.

---

## 9. Support runner contract

Stage 0 support tools:

```text
compiler/test/CodeGen/VC4/Support/check_vc4_test_result.py
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh
```

### 9.1 Result checker

`check_vc4_test_result.py` parses the last `VC4_TEST_RESULT` line in a log and compares it to `expected.json`.

Self-test:

```bash
python3 compiler/test/CodeGen/VC4/Support/check_vc4_test_result.py --self-test
```

### 9.2 Hardware runner

`run_hardware_test.sh` runs one side of a hardware test directory.

Reference side:

```bash
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
  compiler/test/CodeGen/VC4/Hardware/Run/<test-name> \
  reference
```

Candidate side, once codegen exists:

```bash
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
  compiler/test/CodeGen/VC4/Hardware/Run/<test-name> \
  candidate
```

The runner performs:

1. validate `<test-root>/input.mlir`,
2. validate `<test-root>/expected.json`,
3. validate `<test-root>/<side>/run.sh`,
4. power cycle unless `VC4_SKIP_POWER_CYCLE=1`,
5. sleep after power cycle,
6. run `bash run.sh` inside the selected side directory,
7. tee output to `<side>/run.log`,
8. call `check_vc4_test_result.py <test-root>/expected.json <side>/run.log`.

Environment variables:

```bash
VC4_PI_POWER_CYCLE_CMD='uhubctl -l 0-1 -a cycle'
VC4_PI_POWER_CYCLE_SLEEP_SEC=1
VC4_SKIP_POWER_CYCLE=0
VC4_RUN_SH_MAX_ATTEMPTS=3
```

The runner may retry `bash run.sh` when the log contains the known transient serial failure:

```text
tty-USB read() returned 0 bytes.  r/pi not responding [reboot it?]
```

Individual test `run.sh` scripts must not power-cycle the Pi.

---

## 10. Catalog contract

The catalog lives at:

```text
compiler/test/CodeGen/VC4/catalog.json
```

The catalog is factual, not aspirational.

A test may enter the catalog only when all of these are true:

1. `input.mlir` exists.
2. `input.mlir` has a valid lit `RUN:` line.
3. The local `vc4-opt` verifier run passes.
4. The reference bundle exists.
5. `expected.json` exists.
6. The reference run was executed on hardware.
7. `Support/check_vc4_test_result.py expected.json reference/run.log` passed.
8. The test README explains what is being checked.
9. Generated build artifacts have been removed or ignored.
10. The test does not claim candidate/codegen coverage until candidate-side execution exists.

Recommended catalog entry shape:

```json
{
  "implemented_tests": [
    {
      "id": "hardware-run-minimal-thrend",
      "name": "minimal_thrend",
      "kind": "hardware-run",
      "test_dir": "compiler/test/CodeGen/VC4/Hardware/Run/minimal_thrend",
      "input_path": "compiler/test/CodeGen/VC4/Hardware/Run/minimal_thrend/input.mlir",
      "reference_dir": "compiler/test/CodeGen/VC4/Hardware/Run/minimal_thrend/reference",
      "expected_path": "compiler/test/CodeGen/VC4/Hardware/Run/minimal_thrend/expected.json",
      "run_command": "compiler/test/CodeGen/VC4/Support/run_hardware_test.sh compiler/test/CodeGen/VC4/Hardware/Run/minimal_thrend reference",
      "requires_hardware": true,
      "candidate_enabled": false,
      "notes": "Minimal QPU user-program launch/completion test: thread end plus delay slots, no memory output."
    }
  ]
}
```

Do not add future tests to `catalog.json`. Planned tests belong in `compiler/docs/codegen/test-backlog.md`.

---

## 11. Backlog contract

Planned tests live in:

```text
compiler/docs/codegen/test-backlog.md
```

Backlog entries may describe future work but are not coverage.

A backlog entry should say:

- proposed test name,
- feature covered,
- expected `input.mlir` shape,
- expected reference bundle shape,
- expected semantic oracle,
- prerequisites,
- graduation criteria.

Graduation criteria always include: files exist, local MLIR verification passes, reference run passes on hardware, catalog entry added.

---

## 12. Initial hardware-run sequence

The initial hardware-run corpus should grow incrementally. Do not jump directly to a large SAXPY import before smaller hardware facts are stable.

Locked initial sequence:

1. `minimal_thrend`
   - Minimal QPU user-program launch/completion.
   - Qasm: `thrend` plus two delay slots.
   - Oracle: all active QPUs completed.
2. `memory_output`
   - First output-producing kernel.
   - Writes a deterministic value/pattern to memory.
   - Proves launcher memory allocation/copyback and output oracle flow.
3. `read_nop_write`
   - Reads input, performs no meaningful arithmetic, writes output.
   - Proves input + output data movement with minimal compute.
4. `saxpy_reference`
   - Rich reference modeled on the known-good SAXPY bundle.
   - Proves arithmetic + memory movement + work distribution + semantic launcher API.

The SAXPY reference remains the model for public/physical ABI separation, but it is not the first test and it must not freeze SAXPY-specific restrictions as global backend rules.

---

## 13. SAXPY reference test contract

When imported, SAXPY should live at:

```text
compiler/test/CodeGen/VC4/Hardware/Run/saxpy_reference/
```

Required shape:

```text
saxpy_reference/
  README.md
  input.mlir
  expected.json
  reference/
    .gitignore
    Makefile
    run.sh
    3-test-saxpy.c
    mailbox.c
    mailbox.h
    saxpy.qasm
    saxpy_launch.c
    saxpy_launch.h
  share/
    vc4tmpl/template.h
    vc4inc/vc4.qinc
```

`reference/3-test-saxpy.c` must print:

```text
VC4_TEST_RESULT name=saxpy_reference status=PASS mismatches=0 max_abs_diff=<value> n=<N> qpus=<activeQpus> lanes=16
```

`expected.json` should require:

```json
{
  "name": "saxpy_reference",
  "status": "PASS",
  "required": {
    "mismatches": 0
  },
  "float_max": {
    "max_abs_diff": 0.0001
  }
}
```

### 13.1 SAXPY `input.mlir` intent

The SAXPY `input.mlir` should describe the same computation and physical ABI as the reference qasm/launcher.

Semantic operation:

```text
for i in 0..n:
  y[i] = a * x[i] + y[i]
```

Public launcher API:

```c
int saxpy_launch(struct vc4_runtime *rt, float *x, float *y, float a, uint32_t n);
```

Reference physical uniform order per QPU:

```text
x base address
y base address
alpha
n
qpu_id
num_qpus
```

Reference work distribution:

```text
base_element = qpu_id * 16
stride_elements = num_qpus * 16
```

Reference tail policy:

```text
exact_multiple
```

The current reference launcher rejects `n % laneWidth != 0`; this must be represented as this test's policy only, not as a global backend policy.

The device body should be a scheduled QPU sink function equivalent to `reference/saxpy.qasm`.

---

## 14. Assembler-only and litmus tests

Not every hardware-related test is a full reference/candidate codegen semantic pair.

### 14.1 Assembler qualification

Assembler qualification tests verify that the chosen assembler accepts or rejects a specific qasm subset.

Directory shape:

```text
compiler/test/CodeGen/VC4/Hardware/AssemblerQual/<test-name>/
  README.md
  run.sh
  expected.json
  <case>.qasm
```

These tests may not have `input.mlir`, because they qualify assembler behavior rather than codegen semantics. If they later become codegen tests, they should graduate to `Hardware/Run`.

### 14.2 Hardware litmus

Hardware litmus tests resolve hardware facts, such as branch register-source behavior or TMU FIFO depth.

Directory shape:

```text
compiler/test/CodeGen/VC4/Hardware/Litmus/<test-name>/
  README.md
  run.sh
  expected.json
  <test>.qasm
  <test harness files>
```

Litmus tests may not have a candidate side. They are hardware truth-acquisition tests, not ordinary codegen semantic tests.

---

## 15. What must not happen

The following are explicitly forbidden by this contract:

1. Do not add future tests to `catalog.json`.
2. Do not add broad `XFAIL: *` placeholders for missing future compiler passes.
3. Do not claim codegen coverage before candidate-side generation exists.
4. Do not compare generated qasm character-for-character against reference qasm for semantic tests unless exact qasm text is explicitly the contract.
5. Do not require unstable timing/speedup values in `expected.json`.
6. Do not commit transient run outputs or generated build products unless explicitly documented.
7. Do not power-cycle the Pi inside individual test `run.sh` scripts.
8. Do not expose `qpu_id`, `num_qpus`, or raw uniform internals in public launcher APIs unless a test explicitly exists to check such a low-level API.
9. Do not freeze SAXPY-specific exact-multiple tail policy as a global codegen rule.
10. Do not let Codex invent large batches of catalog entries without running them.
11. Do not place an `input.mlir` under `compiler/test` without a `RUN:` line.
12. Do not rely on machine-local absolute vc4asm include/template paths in checked-in tests.

---

## 16. Minimal acceptance checklist for adding a hardware-run test

Before adding a test to `catalog.json`, verify:

```bash
# 1. Files exist.
find compiler/test/CodeGen/VC4/Hardware/Run/<test-name> -maxdepth 3 -type f | sort

# 2. MLIR input exists and has a RUN line.
test -f compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/input.mlir
sed -n '1,5p' compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/input.mlir

# 3. Expected oracle exists.
test -f compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/expected.json

# 4. Reference runner exists.
test -d compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/reference
test -x compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/reference/run.sh

# 5. vc4asm support paths exist if vc4asm -c is used.
test -f compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/share/vc4tmpl/template.h

# 6. Local MLIR verification passes.
compiler/build/bin/vc4-opt \
  compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/input.mlir \
  --vc4-verify-emit-contract \
  --vc4-verify-scheduled-hardware-rules \
  --vc4-verify-scheduled-adjacent-hazards \
  --vc4-verify-scheduled-io-spacing \
  --vc4-verify-scheduled-peripheral-accesses \
  -o /dev/null

# 7. Full local lit suite passes.
cmake --build compiler/build --target check-vc4

# 8. Run reference side on hardware.
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
  compiler/test/CodeGen/VC4/Hardware/Run/<test-name> \
  reference

# 9. Confirm result line.
grep 'VC4_TEST_RESULT' compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/reference/run.log

# 10. Clean transient build outputs before commit.
rm -rf compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/reference/objs
rm -f compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/reference/*.elf
rm -f compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/reference/*.bin
rm -f compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/reference/*.list
rm -f compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/reference/*shader.c
rm -f compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/reference/*shader.h
rm -f compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/reference/run.log

# 11. Confirm generated files are ignored and not staged.
git status --short
git status --ignored --short compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/reference | sed -n '1,120p'
```

Only after these pass should the test be considered implemented.

---

## 17. Summary

The locked VC4 codegen hardware-test contract is:

```text
input.mlir + trusted reference qasm/c/h bundle + hardware semantic oracle
```

Now:

```text
Run reference bundle on hardware -> check expected.json.
```

Later:

```text
Generate candidate bundle from input.mlir -> run on hardware -> compare to the same semantic oracle/reference result.
```

This contract gives future codegen implementation agents a real target while avoiding brittle text overfitting. It also keeps the corpus honest: only materialized, locally verified, hardware-checked tests enter the implemented catalog.
