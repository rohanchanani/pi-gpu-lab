# VC4 CodeGen Hardware Ground-Truth Test Contract

Status: **locked contract for VC4 codegen ground-truth tests**  
Audience: VC4 backend maintainers and coding agents authoring tests  
Applies to: `compiler/test/CodeGen/VC4/**` and `compiler/docs/codegen/test-backlog.md`

This document supersedes earlier “hardware-only reference” drafts. The locked test model is now:

1. Each test has an **input MLIR program** in the final-stage `vc4` dialect form that the code generator is expected to consume.
2. Each test has a **trusted reference bundle** that computes the same semantics on real Raspberry Pi VC4 hardware:
   - `qasm`
   - launcher `.c`
   - launcher `.h`
   - bare-metal test harness and build/run support
3. Before codegen exists, tests validate only the **reference/ground-truth side**.
4. After codegen exists, the same tests also generate a **candidate bundle** from `input.mlir`, build it, run it on hardware, and compare the candidate result with the reference result / semantic oracle.

The goal is not to prove that generated qasm text is character-for-character identical to the reference qasm. The goal is to prove that codegen from the given `vc4` MLIR produces a deployable bundle that computes the same result on hardware.

---

## 1. Hardware and project basis

The VC4 backend targets QPU user programs. A test contract for final codegen must therefore be grounded in actual QPU launch and execution behavior, not only in local text diffs.

Relevant VC4 hardware facts that shape this contract:

- QPUs are 16-way SIMD processors.
- General-purpose user programs are queued through the V3D QPU scheduler user-program request interface. The scheduler receives a program address, uniforms address, uniforms length, and exposes request/completion state.
- Uniforms are a sequential stream: reading the uniform register consumes the next 32-bit word and auto-increments the uniform pointer.
- Program termination uses a thread-end/program-end signal and two following delay-slot instructions.
- TMU, SFU, VPM/VDR/VDW, semaphore, mutex, branch, and thread-switch behavior have real placement and hazard constraints that must ultimately be validated on hardware or explicitly categorized as assembler-only / litmus-only until hardware execution exists.

The practical conclusion is:

> For runnable examples, **real Raspberry Pi hardware execution is the gold standard**. Local checks can verify shape, syntax, and buildability, but semantic correctness is established by the hardware result line.

---

## 2. Test phases

Every test has up to two runnable sides.

### 2.1 Reference / ground-truth side

This side exists first.

It contains a trusted, self-contained bundle and harness that can run on the Pi without the code generator. This bundle may be hand-authored, imported from a known-good reference, or otherwise explicitly trusted by maintainers.

The reference side proves:

- the semantic behavior expected from the test,
- the runtime/build/harness assumptions,
- the launcher ABI shape for the example,
- any relevant hardware behavior exercised by the kernel.

The reference side is what we create now.

### 2.2 Candidate / generated-code side

This side exists later, after codegen exists.

The candidate side is produced from `input.mlir` by the VC4 code generator. It must produce a source bundle:

- `kernel.qasm`
- `kernel_launch.c`
- `kernel_launch.h`

plus whatever local test harness glue is needed to run it through the same Pi flow.

The candidate side passes only if its semantic result matches the reference side.

---

## 3. Directory shape for implemented hardware-run tests

Each implemented hardware-run test must live under:

```text
compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/
```

Required shape:

```text
compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/
  README.md
  input.mlir
  expected.json
  run.sh

  reference/
    README.md                    # optional but strongly recommended
    Makefile
    run.sh
    3-test-<test-name>.c
    mailbox.c
    mailbox.h
    <kernel>.qasm
    <kernel>_launch.c
    <kernel>_launch.h

  candidate/
    README.md                    # placeholder allowed before codegen exists
```

Optional files:

```text
  input.json                     # deterministic host-side input description, if useful
  reference_result.golden.json   # optional captured reference result if stable
  notes.md                       # extra hardware notes, if needed
```

Generated/transient files must not be committed unless the test README explicitly justifies them as source-of-truth artifacts. Normally, do **not** commit:

```text
*.o
*.elf
*.bin
*.list
*shader.c
*shader.h
run.log
candidate/generated/*
```

The root `run.sh` is the compatibility entry point for `Support/run_hardware_test.sh`. During the ground-truth phase, root `run.sh` should run the reference side. Later, the test may grow a separate candidate runner, but root `run.sh` must continue to be understandable and documented.

Recommended root `run.sh` shape during ground-truth phase:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/reference"
bash run.sh
```

The support wrapper performs the power cycle and log checking; individual test `run.sh` scripts must not power-cycle the Pi themselves.

---

## 4. The `input.mlir` contract

`input.mlir` is the MLIR program that the future code generator should consume.

It must describe the same kernel/launch semantics as the reference bundle. It is not merely documentation; it is the compiler input for the future candidate side.

### 4.1 Required properties

`input.mlir` must:

1. Use the `vc4` dialect.
2. Represent the final stage before code generation.
3. Be parseable by the current or intended `vc4` tooling for that stage.
4. Identify the launchable QPU kernel.
5. Carry or reference the launcher ABI metadata needed to generate `launcher.c` and `launcher.h`.
6. Make the physical uniform stream layout explicit, either directly in the IR or through stable codegen metadata.
7. Make the tail policy explicit.
8. Match the semantic behavior of the reference bundle.

### 4.2 Final-stage device body

For v1 codegen, the device side should normally be a scheduled QPU sink function:

```mlir
vc4.func @kernel_name() attributes {
  domain = #vc4.execution_domain<qpu>,
  form = #vc4.function_form<scheduled>,
  kernel,
  threading = #vc4.threading_mode<single>
} {
  // vc4.qpu.bundle / vc4.qpu.ldi / vc4.qpu.sema / vc4.qpu.branch only.
}
```

The existing dialect already has a clear scheduled sink subset for final qasm emission:

- `vc4.qpu.bundle`
- `vc4.qpu.ldi`
- `vc4.qpu.sema`
- `vc4.qpu.branch`

Final qasm emission should consume only this scheduled sink subset in v1.

### 4.3 Launcher ABI metadata

The qasm body alone is not enough to generate the launcher. The launcher needs semantic argument and ABI metadata:

- public launcher function name,
- public semantic arguments,
- C types,
- buffer directions,
- scalar packing,
- physical uniform stream order,
- execution builtin policy,
- active-QPU policy source,
- tail policy,
- whether candidate code object lifetime is persistent or copied per call.

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

### 4.4 Current dialect gap

The existing `vc4` dialect has good scheduled QPU sink operations and function form/domain/threading attributes. However, it does not yet have a fully formal first-class codegen ABI metadata schema for launcher arguments and physical uniform packing.

That is acceptable for the test contract, but it must be called out clearly:

- The **qasm side** of a SAXPY-like final-stage `input.mlir` maps naturally to a scheduled QPU sink function.
- The **launcher side** needs additional codegen metadata that should be formalized early in the codegen milestone.
- Until that metadata is formalized, test `input.mlir` files may use clearly marked provisional `vc4.codegen.*` generic attributes. Those provisional attributes are part of the test contract only if documented in the test README.

This is not a reason to abandon the dialect. It is a precise small codegen-sanity gap: the dialect needs or permits explicit ABI metadata for bundle emission.

---

## 5. Reference bundle contract

The reference bundle is a trusted implementation of the same semantics as `input.mlir`.

Required files in `reference/`:

```text
reference/
  Makefile
  run.sh
  3-test-<test-name>.c
  mailbox.c
  mailbox.h
  <kernel>.qasm
  <kernel>_launch.c
  <kernel>_launch.h
```

For examples modeled on SAXPY, `run.sh` should:

1. assemble qasm with `vc4asm`,
2. generate derived shader C/H files,
3. build the bare-metal test using the local Makefile,
4. boot/run the Pi with `pi-install` or the configured equivalent,
5. stream serial output to stdout.

The reference launcher should expose only semantic arguments plus a runtime handle. It must not expose:

- `qpu_id`,
- `num_qpus`,
- raw uniform structs,
- hardware addresses,
- V3D scheduler registers,
- internal per-QPU uniform arrays.

Those details belong inside the launcher/runtime boundary.

---

## 6. `expected.json` contract

Every hardware-run test root must contain:

```text
expected.json
```

The expected JSON is the semantic oracle for the test. It is intentionally small and stable.

Example:

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
- fields under `required` are exact comparisons,
- fields under `float_max` require `abs(actual) <= limit`.

Do not include unstable values such as execution time, speedup, serial port names, or build paths in required checks.

---

## 7. `VC4_TEST_RESULT` contract

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
6. Timing values may be printed elsewhere but should not be required by `expected.json` unless the test is explicitly about timing.

Example:

```text
VC4_TEST_RESULT name=saxpy_reference status=PASS mismatches=0 max_abs_diff=0.0 n=32768 qpus=12 lanes=16
```

---

## 8. Candidate-side contract after codegen exists

After the code generator exists, each hardware-run test grows a candidate path.

Future candidate workflow:

1. Read `input.mlir`.
2. Run codegen to produce candidate:
   - `candidate/<kernel>.qasm`
   - `candidate/<kernel>_launch.c`
   - `candidate/<kernel>_launch.h`
3. Build candidate with the same or analogous harness.
4. Run candidate on hardware.
5. Capture a candidate `VC4_TEST_RESULT`.
6. Compare candidate result against the same `expected.json`, and where appropriate compare against a captured reference result.

The candidate bundle does **not** need to match the reference bundle textually.

The following differences may be valid:

- different but legal instruction scheduling,
- different register allocation,
- different labels,
- different temporary layout,
- different but semantically equivalent launcher implementation,
- persistent code-object handling instead of per-launch shader copying,
- tail-safe handling where the reference was exact-multiple-only, if the MLIR input says so.

The following differences are not valid:

- changed public semantic API without metadata justification,
- missing or reordered uniform fields relative to the declared ABI,
- public exposure of execution builtins unless the test explicitly requires it,
- failing to run on hardware,
- different output semantics,
- violating hardware scheduling rules,
- relying on host interrupts when the test forbids them.

---

## 9. Catalog contract

The catalog lives at:

```text
compiler/test/CodeGen/VC4/catalog.json
```

Stage 0 catalog:

```json
{
  "implemented_tests": []
}
```

A test may enter the catalog only when all of these are true:

1. `input.mlir` exists.
2. Reference bundle exists.
3. `expected.json` exists.
4. Root `run.sh` exists.
5. The reference run was executed on hardware.
6. `Support/check_vc4_test_result.py expected.json run.log` passed.
7. The test README explains what is being checked.
8. The test does not claim candidate/codegen coverage until candidate-side execution exists.

Recommended catalog entry shape:

```json
{
  "implemented_tests": [
    {
      "id": "hardware-run-saxpy-reference",
      "name": "saxpy_reference",
      "kind": "hardware-run",
      "input_mlir": "compiler/test/CodeGen/VC4/Hardware/Run/saxpy_reference/input.mlir",
      "test_dir": "compiler/test/CodeGen/VC4/Hardware/Run/saxpy_reference",
      "reference_dir": "compiler/test/CodeGen/VC4/Hardware/Run/saxpy_reference/reference",
      "expected_path": "compiler/test/CodeGen/VC4/Hardware/Run/saxpy_reference/expected.json",
      "run_command": "compiler/test/CodeGen/VC4/Support/run_hardware_test.sh compiler/test/CodeGen/VC4/Hardware/Run/saxpy_reference",
      "requires_hardware": true,
      "reference_passed": true,
      "candidate_enabled": false,
      "notes": "Known-good SAXPY reference bundle paired with final-stage vc4 input."
    }
  ]
}
```

The catalog is factual, not aspirational. Future tests stay in the backlog until implemented and passed.

---

## 10. Backlog contract

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

Graduation criteria always include: files exist, reference run passes on hardware, catalog entry added.

---

## 11. Support tools

Stage 0 support tools:

```text
compiler/test/CodeGen/VC4/Support/check_vc4_test_result.py
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh
```

### 11.1 Result checker

`check_vc4_test_result.py` parses the last `VC4_TEST_RESULT` line in a log and compares it to `expected.json`.

Self-test:

```bash
python3 compiler/test/CodeGen/VC4/Support/check_vc4_test_result.py --self-test
```

### 11.2 Hardware runner

`run_hardware_test.sh` runs a hardware test directory.

Example:

```bash
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
  compiler/test/CodeGen/VC4/Hardware/Run/saxpy_reference
```

It performs:

1. power cycle unless `VC4_SKIP_POWER_CYCLE=1`,
2. sleep,
3. run root `run.sh`,
4. tee output to `run.log`,
5. call `check_vc4_test_result.py expected.json run.log`.

Environment variables:

```bash
VC4_PI_POWER_CYCLE_CMD='uhubctl -l 0-1 -a cycle'
VC4_PI_POWER_CYCLE_SLEEP_SEC=4
VC4_SKIP_POWER_CYCLE=0
```

---

## 12. SAXPY reference test contract

The first implemented test should be:

```text
compiler/test/CodeGen/VC4/Hardware/Run/saxpy_reference/
```

It should import the known-good reference bundle from:

```text
old_compiler/reference/saxpy/
```

Required shape:

```text
saxpy_reference/
  README.md
  input.mlir
  expected.json
  run.sh
  reference/
    Makefile
    run.sh
    3-test-saxpy.c
    mailbox.c
    mailbox.h
    saxpy.qasm
    saxpy_launch.c
    saxpy_launch.h
  candidate/
    README.md
```

`reference/3-test-saxpy.c` must be modified from the original reference to print:

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

### 12.1 SAXPY `input.mlir` intent

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
exact_multiple_of_16
```

The current reference launcher rejects `n % laneWidth != 0`; this must be represented as this test's policy only, not as a global backend policy.

The device body can be represented as a scheduled QPU sink function equivalent to `reference/saxpy.qasm`.

A future dialect/codegen metadata milestone should formalize the launcher ABI attributes used by this input. Until then, the test may use provisional `vc4.codegen.*` attributes with a README note that they are codegen metadata, not hardware instructions.

---

## 13. Provisional SAXPY `input.mlir` sketch

This sketch is not a replacement for the full imported test file, but it captures the intended shape.

```mlir
vc4.module @saxpy_reference {
  vc4.func @saxpy_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,

    // Provisional codegen metadata. This should be formalized in the codegen
    // metadata milestone before the emitter consumes it.
    vc4.codegen.public_launcher = "saxpy_launch",
    vc4.codegen.tail_policy = "exact_multiple_of_16",
    vc4.codegen.uniform_abi = [
      "x:buffer:in:f32",
      "y:buffer:inout:f32",
      "a:scalar:f32",
      "n:scalar:u32",
      "builtin.qpu_id:u32",
      "builtin.num_qpus:u32"
    ],
    vc4.codegen.work_distribution = "base=qpu_id*16,stride=num_qpus*16"
  } {
    // The full scheduled sink body should be equivalent to reference/saxpy.qasm.
    // It will consist only of vc4.qpu.bundle, vc4.qpu.ldi, vc4.qpu.sema,
    // and vc4.qpu.branch once translated fully into current vc4 sink syntax.
  }
}
```

When the SAXPY test is imported, `input.mlir` should either contain the full scheduled sink body or explicitly state that the full body will be filled in by the codegen-metadata milestone before candidate-side execution is enabled. A cataloged candidate-enabled test must have a complete input body.

---

## 14. Assembler-only and litmus tests

Not every hardware-related test is a full reference/candidate pair.

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

---

## 16. Minimal acceptance checklist for adding a hardware-run test

Before adding a test to `catalog.json`, verify:

```bash
# 1. Files exist.
find compiler/test/CodeGen/VC4/Hardware/Run/<test-name> -maxdepth 3 -type f | sort

# 2. MLIR input exists.
test -f compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/input.mlir

# 3. Expected oracle exists.
test -f compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/expected.json

# 4. Runner exists.
test -x compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/run.sh

# 5. Reference bundle exists.
test -d compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/reference
test -x compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/reference/run.sh

# 6. Run on hardware.
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
  compiler/test/CodeGen/VC4/Hardware/Run/<test-name>

# 7. Confirm result line.
grep 'VC4_TEST_RESULT' compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/run.log

# 8. Confirm catalog entry points to real files.
cat compiler/test/CodeGen/VC4/catalog.json
```

Only after these pass should the test be considered implemented.

---

## 17. Summary

The locked VC4 codegen test contract is:

```text
input.mlir  +  trusted reference qasm/c/h bundle  +  hardware semantic oracle
```

Now:

```text
Run reference bundle on hardware -> check expected.json.
```

Later:

```text
Generate candidate bundle from input.mlir -> run on hardware -> compare to the same semantic oracle/reference result.
```

This contract gives future codegen implementation agents a real target while avoiding brittle text overfitting. It also keeps the corpus honest: only materialized, hardware-checked tests enter the implemented catalog.
