# AGENTS.md — VC4 Compiler Project Guidance

This file is guidance for Codex/AI agents working in `rohanchanani/pi-gpu-lab` on the `compiler` branch. It records the project invariants, verification doctrine, hardware discipline, debugging expectations, and phase-package style for the VC4 compiler stack.

Treat this file as a standing contract. Later phase prompts may add narrower instructions, but they must not violate the hard rules below.

---

## 1. Project purpose and current stack

This project is an MLIR-based compiler/backend for Raspberry Pi VideoCore IV (VC4) QPUs. It is correctness-first and hardware-backed. The goal is not to produce pretty IR only; the goal is to lower real kernels all the way to real VC4 hardware and prove device results against host oracles.

The locked post-P13 stack boundary is:

```text
standard MLIR value layer
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime
  -> real VC4 hardware
```

The future full upper stack is:

```text
Triton / TTIR
  -> standard MLIR value layer
       func + tiny vc4value + vector + memref + arith + math + scf/cf
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime
  -> real VC4 hardware
```

`vc4kernel` is the closed VC4 target-kernel planning contract above SSAVC4. It may use MLIR vector types such as `vector<16xi32>` and `vector<16xf32>` as target fragment carrier types, but verified `vc4kernel` kernels must not contain `vector` dialect operations or producer dialect operations.

The standard value layer is producer-facing. It expresses ordinary value semantics using `func`, tiny `vc4value`, `vector`, `memref`, `arith`, `math`, `scf`, and `cf`. It must not expose VC4 hardware data-path operations directly.

---

## 2. Absolute hard rules

These rules are non-negotiable.

### 2.1 Layering rules

- No direct `VC4KernelToVC4` path.
- No direct lowering from value IR, TTIR, or Triton to scheduled `vc4`.
- No direct lowering from Triton/TTIR to `vc4kernel` as the first producer integration.
- No VC4Tile resurrection, compatibility layer, or renamed tile DSL.
- No producer dialect operations inside verified `vc4kernel`.
- No source-authored `ssavc4` or scheduled `vc4` pretending to be value-layer output.
- `vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware` remains the required lower-half path.

### 2.2 `vc4value` scope rules

`vc4value` must remain tiny launch/policy plumbing.

Allowed initial concepts:

```text
vc4value.program_id(axis)   : index
vc4value.num_programs(axis) : index
vc4value.kernel metadata
vc4value.grid_rank metadata
math/profile/policy metadata if explicitly phased
```

Forbidden in `vc4value` unless a future phase explicitly reopens the design:

```text
vc4value.load
vc4value.store
vc4value.tile
vc4value.vpm
vc4value.tmu
vc4value.vdr
vc4value.vdw
vc4value.fragment
vc4value.barrier
vc4value.lane_id, unless vector.step is proven inadequate
```

The value layer should expose `memref`/`vector` semantics; the planner chooses TMU vs VDR/VPM vs VDW.

### 2.3 Verification and anti-cheat rules

Never:

- weaken verifier diagnostics;
- weaken hardware oracles;
- remove sentinels;
- reduce adversarial dimensions just to pass;
- special-case fixture names, public names, candidate names, paths, status strings, or generated-output paths;
- substitute expected generated output for real lowering;
- reshape natural kernels to paper over compiler/lower-half bugs;
- hide semantics in comments only;
- claim hardware support without hardware proof;
- continue after a real blocker and call it success.

Use `FAILURE` for real blockers, ambiguous repo state, required command failures, hardware mismatches/timeouts, or any situation where passing would require cheating.

---

## 3. VC4 hardware facts that must shape compiler work

The VideoCore IV QPU is effectively a 16-way 32-bit SIMD processor. It has add and multiply ALU pipes, register-file hazards, special accumulators, external hardware units, and strict scheduling restrictions. Keep these facts in mind when designing or debugging compiler changes.

Important hardware facts:

- QPU SIMD width is 16 lanes.
- QPU code is scheduled through a central QPU scheduler for general-purpose programs.
- Uniforms are a stream of 32-bit values from memory.
- TMU supports direct 32-bit general-memory lookups; bottom address bits are ignored for direct-address 32-bit lookups.
- SFU supports approximate target operations such as recip, recipsqrt, exp, and log through shared slice hardware; results arrive in `r4` after required latency.
- VPM is a 2D array from the QPU point of view, 16 32-bit words wide, with horizontal/vertical and 8/16/32-bit access modes.
- VDR loads global memory into VPM; VDW stores VPM/global output; VDR and VDW are not symmetric.
- VPM QPU read/write, VDR DMA, and VDW DMA have separate setup encodings and constraints.
- QPU programs have branch delay slots and thread-end restrictions.
- External hardware accesses can stall.
- In most cases, only one closely coupled peripheral access may occur in a single instruction.
- Performance counters exist and should be used later for observability, not as a replacement for correctness oracles.

Do not infer that a hardware-looking thing is legal just because a register field exists. Check the locked surface matrix, verifier, conversion, and hardware fixtures.

---

## 4. Locked VC4Kernel final surface facts

Post-P13 VC4Kernel Surface v2 is locked. All `vc4kernel` work must preserve that contract.

Accepted/hardware-proven feature families include:

- kernel ABI, return, program id, num programs, warp identity, lane identity;
- full/empty/tail/rect predicates and boolean predicate ops;
- fragment const, bitcast, splat, select;
- add-pipe and mul-pipe fragment ALU opcodes;
- i32 and finite-policy f32 comparisons;
- i32 and finite-tree-policy f32 reductions;
- scalar arithmetic/control/address subset;
- TMU safe-offset inactive loads;
- VDW register-fragment inactive-preserve stores;
- VPM allocation and QPU read/write for accepted w32/subword/dynamic coordinates/selectors;
- VDR-to-VPM for accepted w32/subword/dynamic coordinates/selectors;
- VDW-from-VPM for accepted w32/subword/dynamic coordinates/selectors;
- pack/unpack;
- f16 storage conversion plus f32 compute;
- approximate SFU policy;
- dynamic rotate and rotate-derived composites;
- barriers, semaphores, cooperative metadata;
- runtime resource metadata and libpi descriptors;
- artifact emission, manifest JSON, generated C/QASM.

Deterministic rejects include:

- producer dialect operations inside VC4Kernel;
- removed fragment special spellings such as legacy `fragment_add`, `fragment_sub`, `fragment_mul`, `fragment_shl`;
- old TMU signatures and inferred safe addresses;
- sparse VDW stores;
- DMA laned subword modes;
- dynamic orientation, dynamic width, or dynamic subword mode attrs;
- vector or lane-varying coordinate operands;
- horizontal w32 dynamic word-X when not meaningful;
- arbitrary shuffle/permutation at the `vc4kernel` level;
- native f16 arithmetic;
- native bf16/fp8 arithmetic/conversion;
- exact/default math silently lowering to SFU.

When in doubt, inspect:

```text
compiler/docs/vc4kernel_surface_v2_final_lock.md
compiler/docs/vc4kernel_surface_v2_support_matrix.json
compiler/docs/codegen/vc4kernel_dialect_strict_specification.md
compiler/docs/vc4kernel_mixed_acceptance_policy.md
```

---

## 5. Value layer abstraction boundary

The value layer is a target-profiled standard MLIR subset. It is not TTIR, and it is not `vc4kernel`.

Allowed initial value-layer dialect family:

```text
builtin
func
vc4value
vector
memref
arith
math
scf
cf
```

Forbidden initially:

```text
tt
ttg
gpu
linalg
tensor
nvgpu
nvvm
rocdl
spirv
iree
stablehlo
mhlo
vc4kernel mixed into value input
ssavc4
vc4
```

The value layer may be broader than the first executable slice. Distinguish status carefully:

```text
surface_admissible
lowerable_v1
staged_split_required
staged_contract_required
staged_memory_planning_required
supported_composite
supported_emulated
supported_with_policy_caveat
temporary_reject_not_implemented
permanent_reject_with_proof
deterministic_reject
internal_only
```

`vector<16xT>` is the V1 lowerable hardware fragment shape, not the whole value abstraction.

Recommended vector policy:

- `vector<16xT>`: V1 lowerable fragment shape where element type and op are supported.
- fixed rank-1 `vector<NxT>` where `N != 16`: surface-admissible/staged; lower later by splitting into 16-lane fragments plus tails.
- fixed rank-2 vectors: surface-admissible/staged for contracts, transpose, and tile planning.
- scalable vectors: reject.
- initial `tensor`/`linalg`: reject; may be a later producer layer.

Element type policy:

- `i1`: masks/predicates.
- `index`: value-layer address/program-id math; lower to i32 when ABI/lowering requires.
- `i8`, `i16`: surface/storage-admissible for subword memory paths; arithmetic support is separate.
- `f16`: storage-admissible through f16 storage conversion plus f32 compute; native f16 arithmetic is rejected.
- `i32`, `f32`: first executable arithmetic carriers.
- `bf16`, `fp8`, `f64`, unsupported exotic types: reject unless a future phase explicitly classifies them.

---

## 6. ABI expectations for the value layer

Value kernels use `func.func` plus `vc4value` metadata. Lowering later produces `vc4kernel.kernel`.

Expected public value-layer ABI shape:

```mlir
func.func @kernel(
  %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"},
  %y: memref<?xf32, #vc4value.global> {vc4value.arg_name = "y", vc4value.direction = "out"},
  %n: index {vc4value.arg_name = "n"}
) attributes {vc4value.kernel, vc4value.grid_rank = 1} {
  %pid = vc4value.program_id {axis = 0} : index
  ...
  return
}
```

ABI principles:

- Public kernel args are scalars and ranked global memrefs, not vector/tensor args.
- Memrefs are logical value-layer buffers; lower later to raw i32 base pointer uniforms plus explicit scalar sizes/strides as needed.
- No hidden memref descriptor ABI unless a future phase explicitly implements it.
- `memref.dim` is metadata-only in early phases; do not assume runtime descriptor lowering.
- `program_id` is logical launch-grid identity, not physical QPU identity.
- `num_programs` is launch-grid metadata.
- Phase 5 V1 lowerability is narrower than ABI admissibility.

---

## 7. Hardware verification doctrine

Real hardware tests are the gold standard for executable semantics.

A feature is not executable-verified until a kernel using that feature is lowered all the way to real VC4 hardware, run on the device, and compared against a host oracle with sentinel checks.

Static tests are necessary but insufficient:

```text
lit/parser/verifier tests:
  syntax, diagnostics, conversion shape, rejection policy

hardware tests:
  executable semantics, lower-half behavior, scheduling/runtime correctness
```

### 7.1 Isolation fixtures

For each new executable feature band, add at least three new isolation hardware fixtures unless the phase prompt explicitly justifies a different number.

Isolation fixtures should:

- be designed to break the new feature;
- sweep a meaningful range of sizes, values, masks, tails, alignments, signs, edge cases, and dynamic parameters as applicable;
- use strict host CPU oracles;
- use sentinels before/after output regions;
- prove computation is happening on the device, not in generated host code;
- remain in the repo for future triage.

Isolation fixtures are not the routine cumulative final acceptance suite. They are feature-specific proofs and future debugging tools.

### 7.2 Mixed fixtures

For every phase that adds executable lowering or changes hardware-affecting compiler behavior, add one or two mixed fixtures, or strengthen existing mixed fixtures.

Mixed fixtures should:

- combine many features implemented so far;
- look increasingly like real kernels;
- create feature interactions likely to expose bugs;
- cover every implemented feature in multiple contexts over time;
- use checked output/oracles/sentinels, not decorative IR;
- be part of the routine final acceptance suite.

At first, mixed fixtures may resemble isolation fixtures because the feature set is small. As the value/Triton layer grows, mixed fixtures should evolve toward realistic kernels:

```text
elementwise copy/add/saxpy
masked activation/relu/select
row reductions
softmax-like row kernels
GEMV
GEMM/matmul microkernels
subword/quantized kernels
approx-math activation kernels
cooperative/tiled kernels
```

### 7.3 Final acceptance for hardware phases

At the end of each hardware-affecting phase:

- run the full current mixed suite for that layer;
- run required audits, matrix checkers, lit suites, and `check-vc4`;
- do not rerun every historical isolation fixture as routine final acceptance unless debugging;
- do not skip mixed acceptance because isolated fixtures passed.

### 7.4 Claim discipline

Every mixed fixture `saw_*`, `no_*`, or equivalent claim must be auditable.

Claims must be backed by one of:

```text
CHECKED_OUTPUT
CHECKED_AUDIT
PHASE_GUARD
```

Claims may not be based on:

- dead IR;
- decorative operations;
- broad fixture names;
- status strings;
- generated-output path checks;
- comments;
- unverified assumptions.

Follow the existing `vc4kernel` precedent:

```text
compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_fixture_claims.json
compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/audit_mixed_fixture_claims.py
```

Create analogous value/Triton claim auditing when new upper-layer mixed fixtures appear.

### 7.5 Phase 5 value-layer hardware doctrine

Phase 5 is the first executable standard value-layer slice. Static verifier,
conversion, and lit tests can prove syntax, diagnostics, and lowering shape,
but they do not prove executable value semantics.

For Phase 5 value lowering:

- every executable feature band needs targeted isolation hardware fixtures
  with strict CPU oracles and sentinels;
- routine final acceptance must use value-layer mixed fixtures that combine
  launch identity, lane math, masks, transfer reads/writes, ALU, compare,
  select, TMU safe offsets, and VDW preserve behavior;
- value mixed fixtures must have a manifest and claim audit for every
  `saw_*`, `no_*`, or equivalent claim, following the VC4Kernel mixed fixture
  precedent;
- the value hardware runner must generate fresh candidates from source and
  lower only through `vc4kernel -> ssavc4 -> scheduled vc4`;
- CPU oracles, sentinels, expected JSON, power-cycle discipline, timeout
  policy, and result checking are part of the proof and must not be weakened;
- failed mixed fixtures should be debugged by running the relevant isolation
  fixtures, reducing the mixed kernel, and inspecting value IR, VC4Kernel,
  SSAVC4, scheduled VC4, emitted artifacts, and runtime packing;
- real lower-half bugs in VC4Kernel, SSAVC4, scheduled VC4, artifact emission,
  or runtime must be escalated as blockers unless the current prompt explicitly
  authorizes fixing that layer;
- do not reshape natural value kernels to hide compiler or lower-half bugs.

---

## 8. Hardware runner discipline

Use the existing project hardware runners and policies. Do not invent a parallel runner when an existing one applies.

Known final VC4Kernel mixed acceptance runner:

```bash
VC4_CODEGEN_STATE_ROOT=.vc4_auto/codegen_p13h_final_mixed_acceptance \
  compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/run_mixed_acceptance.sh \
  --manifest compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_acceptance_manifest.json
```

For new value/Triton phases, inspect and follow the existing hardware test layout under:

```text
compiler/test/CodeGen/VC4Kernel/Hardware/
compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/
compiler/test/CodeGen/SSAVC4/
compiler/test/CodeGen/VC4/
```

Hardware requirements:

- generate fresh candidates from source;
- do not use stale `.vc4_auto` bundles;
- use the existing strict runner;
- power-cycle before hardware runs under the project runner policy;
- use a 60s timeout per attempt unless the fixture documents a justified exception;
- treat pre-boot transient failures according to the existing runner policy;
- treat after-boot mismatch, timeout, sentinel failure, or launch failure as a real failure;
- require `VC4_TEST_RESULT status=PASS` and zero mismatch/sentinel/launch-failure fields where present;
- preserve expected JSON, host oracle, sentinels, and result checkers.

If hardware access is not available for a hardware-required phase, stop with `FAILURE`. Do not substitute static tests.

---

## 9. Debugging doctrine

Be an active debugger. Do not just stare at code.

When a mixed fixture fails:

1. Identify every feature involved in the mixed fixture.
2. Run isolation fixtures for those feature bands.
3. If isolation fixtures pass, reduce the mixed fixture:
   - remove one feature at a time;
   - vary sizes, masks, values, tails, dynamic parameters;
   - preserve the failing structure as much as possible.
4. Inspect each lowering layer:
   - value IR / `vc4value` / `vector`;
   - `vc4kernel`;
   - `ssavc4`;
   - scheduled `vc4`;
   - emitted QASM/C/manifest;
   - runtime uniform/resource packing.
5. Instrument hypotheses aggressively:
   - write diagnostic values to output buffers;
   - add temporary debug stores in the current branch;
   - manipulate lower-level IR to test a hypothesis;
   - rerun from intermediate stages where supported;
   - compare host/device intermediate values if feasible.
6. If the bug appears to be in a lower layer than the current phase, stop and report it. Do not paper over it.

You may inspect and locally instrument lower-half code while debugging a higher-layer phase. But if a true lower-half bug is found, report it as a blocker unless the current prompt explicitly authorizes fixing that layer.

Do not reshape kernels to hide compiler/lower-half bugs. GEMV, GEMM, natural loops, spills, branches, VPM reuse, and tails are intentional stressors.

---

## 10. Known lower-half hazards to remember

These bug classes have occurred before. Do not reintroduce them.

### 10.1 VDR/VDW asymmetry

VDR and VDW are not symmetric. VDW source layout, stride fields, preserve behavior, and memory output must be handled separately from VDR destination layout.

### 10.2 Subword selector semantics

Keep these fields distinct:

```text
row
word-X
subword selector
static width
static subword mode
static orientation
```

A dynamic subword selector is a byte/halfword selector inside an accepted packed mode. It is not dynamic width, dynamic subword mode, dynamic orientation, or dynamic layout.

### 10.3 Spill reload coherency

Hidden spill slots written through VDW must not be reloaded through incoherent TMU shortcuts. Prior failures involved spill reloads that needed VDR -> VPM -> QPU VPM read rather than TMU.

### 10.4 Branch/layout accounting

Do not reintroduce manual flattened branch slot mirrors. Use planned/emission-derived accounting. Branch delay slots and thread-end delay slots matter.

### 10.5 Sparse stores

Sparse VDW stores remain rejected unless a future hardware-proven phase adds a legal path. Masked loads and masked stores have different legality.

### 10.6 Exact/default math

Do not silently lower exact/default math to approximate SFU. SFU requires explicit approximate/target policy.

### 10.7 Lane movement

Dynamic rotate and rotate-derived composites are accepted. Arbitrary shuffle/permutation is rejected at `vc4kernel`. Lane broadcast is a composite idiom, not a new first-class `fragment_broadcast_lane` op:

```text
lane_range
  -> compare lane == selected_lane
  -> select value or zero
  -> fragment_reduce add
```

For bit-preserving f32 lane broadcast, use:

```text
fragment_bitcast f32 -> i32
one-hot i32 broadcast composite
fragment_bitcast i32 -> f32
```

---

## 11. Value-to-VC4Kernel lowering principles

The value-to-`vc4kernel` planner is responsible for translating standard value semantics into target execution decisions.

It must:

- split larger fixed vectors into `vector<16>` fragments plus tails;
- map `vector.step` / arange-like idioms to lane identity;
- classify masks as full, empty, tail, rect, sparse, or unknown;
- choose TMU vs VDR/VPM for loads;
- choose VDW preserve paths for stores;
- reject sparse stores unless a legal future path exists;
- lower `vector.reduction` only when numeric policy permits;
- lower `vector.contract` through explicit tiling/planning, not by inventing a tile DSL;
- lower legal broadcast/rotate/shuffle patterns through accepted native or composite idioms;
- preserve f16 storage semantics as storage conversion plus f32 compute;
- keep approximate math explicit;
- lower `scf` to `cf`/block args before verified `vc4kernel`;
- emit only accepted `vc4kernel` operations.

The value layer must not expose:

```text
TMU
VDR
VDW
VPM
physical QPU IDs
VC4 resource metadata as source semantics
VC4Tile-like tile ops
```

---

## 12. Triton/TTIR policy

Triton support comes after the handwritten value path is proven.

Initial architecture:

```text
real Triton source
  -> real emitted TTIR / tt dialect
  -> vc4-triton-import or equivalent importer
  -> standard value-layer MLIR
  -> value-to-vc4kernel
  -> hardware
```

Rules:

- Do not consume TTGIR/NVIDIA-specific IR first.
- Do not parse TTIR with regexes.
- Do not hand-write fake TTIR as the main acceptance path.
- Check in real Triton source examples and emitted TTIR artifacts for reproducibility.
- Keep normal `check-vc4` independent of a hard Triton regeneration dependency when possible.
- The formal support/reject matrix should classify TTIR op/forms; demos should still look like real Triton kernels.

Definition of FULL Triton support:

```text
Every relevant Triton/TTIR construct eventually either:
  A. lowers through the value surface to hardware, or
  B. deterministically rejects with a compelling hardware/semantic proof explaining why VC4 cannot support it.
```

Temporary “not implemented yet” is allowed during staging but must shrink over time. Permanent rejects require proof, not inconvenience.

---

## 13. Phase/package style

Real implementation phases should be split into manageable prompt packages. Do not make one prompt design, implement, hardware-prove, audit, and final-accept a major feature family all at once.

Typical package shape:

```text
README.txt
PhaseNa_..._inventory_design_readonly.txt
PhaseNb_..._first_narrow_implementation.txt
PhaseNc_..._targeted_static_or_hardware_proof.txt
PhaseNd_..._second_feature_or_stress_proof.txt
PhaseNe_..._audit_docs_rejects_migration_lock.txt
PhaseNf_..._mixed_acceptance_fixtures.txt
PhaseNg_..._final_acceptance.txt
```

Each prompt should be a coherent commit-sized step.

Every prompt should include:

- context and phase goal;
- prerequisites/readiness lines from previous phases;
- exact scope and non-goals;
- files likely touched;
- implementation requirements or read-only tasks;
- required tests/commands;
- hardware discipline if applicable;
- forbidden shortcuts;
- commit protocol;
- final `SUCCESS`/`FAILURE` response contract.

Read-only inventory prompts should write reports under `.vc4_auto/...` and should not modify tracked source files.

Implementation prompts should commit only their scoped changes after verification. Do not commit `.vc4_auto` artifacts unless a prompt explicitly says to source-control a generated artifact.

---

## 14. Standard commands and helpers

Common build/check commands:

```bash
ninja -C compiler/build vc4-opt vc4-codegen
ninja -C compiler/build check-vc4
```

Common lit helper:

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

run_if_nonempty_lit() {
  d="$1"
  if [ -d "$d" ] && find "$d" \( -name '*.mlir' -o -name '*.test' \) -print -quit | grep -q .; then
    run_lit -sv "$d"
  else
    echo "skipping missing or empty lit dir: $d"
  fi
}
```

Common VC4Kernel final audits/checks:

```bash
python3 compiler/test/Dialect/VC4Kernel/Support/check_vc4kernel_surface_v2_matrix.py \
  compiler/docs/vc4kernel_surface_v2_support_matrix.json \
  --mode final

python3 compiler/test/Dialect/VC4Kernel/Support/audit_vc4kernel_surface_v2.py \
  --repo-root . \
  --matrix compiler/docs/vc4kernel_surface_v2_support_matrix.json \
  --mode p13-final-surface-lock \
  --phase-lock P13

python3 compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/check_mixed_acceptance_coverage.py \
  --manifest compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_acceptance_manifest.json \
  --repo-root . \
  --mode lock

python3 compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/audit_mixed_fixture_claims.py \
  --claims compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_fixture_claims.json \
  --manifest compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_acceptance_manifest.json \
  --repo-root .
```

Before and after each prompt:

```bash
git status --short --branch
git diff --stat
git diff --check
```

If `compiler/build/build.ninja` is missing, do not improvise a reconfigure unless the phase prompt explicitly permits it. Report the missing build directory.

---

## 15. Current staged roadmap

The staged roadmap after P13 is:

```text
0  Post-P13 rebaseline
1  Full value/Triton taxonomy and target-profile spec
2  Tiny vc4value launch/policy layer
3  Value-surface verifier and audit skeleton
3.5 Value-surface abstraction correction/general fixed-vector staging
4  Kernel wrapper, launch ABI, and memref ABI
5  V1 handwritten value elementwise to hardware
6  Real TTIR inventory and importer skeleton
7  Real TTIR elementwise smoke to hardware
8  Control flow and loop boundary
9  Mask classifier and richer memory legality
10 Memory expansion: gather loads and strided transfers
11 Reductions
12 TTIR reductions smoke
13 Math and SFU policy
14 TTIR math smoke
15 Subword types and pack/unpack
16 TTIR subword smoke
17 Shuffle, rotate, transpose
18 VPM tile transfer planner
19 vector.contract row-fragment GEMM
20 TTIR dot smoke
21 GEMM shape expansion
22 Cooperative block planning
23 VPM pipelining and double buffering
24 Block pointers and tensor descriptors
25 Broader Triton memory profile
26 Frontier-lab-relevant Triton kernel corpus
27 Source-to-hardware driver
28 Full TTIR operation/profile closure
29 Value/Triton mixed acceptance lock
30 Optimization and broadening
```

Near-term focus:

- Phase 5 proves handwritten value elementwise kernels on hardware.
- Phase 6 inventories real TTIR and importer shape.
- Phase 7 proves real Triton elementwise smoke to hardware.

Do not skip Phase 5 and jump to TTIR.

---

## 16. Final response contract for Codex phases

Unless a prompt explicitly says otherwise, final responses must start with exactly one word:

```text
SUCCESS
```

or

```text
FAILURE
```

Then include concise structured facts:

```text
PHASE_RESULT=...
READY_FOR_NEXT_PHASE=YES/NO
COMMIT=<hash or none>
REPORT=<path if any>
TESTS=<summary>
HARDWARE=<summary or not applicable>
BLOCKERS=<none or list>
CAVEATS=<none or list>
```

Do not call something `SUCCESS` if required hardware did not run, required audits failed, the repo has unrelated dirty files, or correctness was achieved by weakening tests or avoiding the natural kernel.

---

## 17. When unsure

If something is ambiguous:

1. Inspect the current repo docs, support matrices, final lock docs, and phase reports.
2. Prefer deterministic reject or `FAILURE` over guessing.
3. Do not broaden scope without explicit instruction.
4. Ask for a smaller remediation package if the problem is outside the current phase.
5. Preserve layer boundaries and hardware proof discipline above all else.

The project succeeds by being honest, hardware-proven, and incremental.

This addendum applies to all future VC4 compiler work. It is intentionally strict because the compiler is now moving from smoke-proven vertical slices into reusable infrastructure.

### A. No temporary architecture

Do not land code that is known to be a temporary, brittle, stringly typed, fixture-specific, or workaround-shaped foundation.

Examples of forbidden foundations:

- semantic decisions based on printed IR/type/attribute substrings;
- fixture-name, test-path, public-name, candidate-name, or status-string special cases;
- Python semantic import paths for accepted TTIR lowering after the C++ importer is locked;
- output IR produced by unlinking/leaking old operations instead of using proper MLIR ownership;
- bypassing an intended layer because the proper layer is difficult;
- accepting “we will clean this up later” for code that future phases must build on.

If the correct long-term design is clear, implement that design even if it is harder. If the correct long-term design is ambiguous, stop with `FAILURE` or `BLOCKED` and name the exact design choices that require user input. Do not force through a short-term design.

### B. Semantic classification must be structural

Compiler semantics must come from typed APIs, operation classes, exact operation names, exact attributes, SSA use-def structure, verified dialect contracts, and documented hardware facts.

Allowed string uses:

- exact operation-name constants when generic MLIR operation handling is deliberate;
- exact attribute-name lookup followed by typed/exact interpretation;
- diagnostics, reports, and debug/provenance output;
- source-name metadata that does not decide semantic roles.

Forbidden semantic string uses:

- parsing printed types to decide lowering behavior;
- parsing printed operations or attributes to decide lowering behavior;
- permissive substring matching such as “contains x” or “contains 0” for semantic decisions;
- using source argument names to decide pointer direction, tail-bound role, memory role, or type legality;
- using comments or documentation as a substitute for verifier/conversion logic and tests.

Every new frontend/value feature must have audits that enforce this boundary where practical.

### C. Use proper MLIR ownership and output construction

Passes must not rely on unlinking, leaking, or leaving old operations in an invalid ownership state.

For importer/conversion passes that replace one IR layer with another, prefer:

- analyze input without mutating it;
- build a clean output representation;
- verify the output boundary;
- commit only after the full conversion succeeds;
- preserve atomic failure behavior;
- destroy or replace old operations using standard MLIR ownership mechanisms.

Do not use `Operation::remove()` or similar unlinking as a long-term solution to avoid erasure/destruction issues. If proper erasure exposes an assertion or ownership bug, stop and fix the ownership/modeling problem.

### D. Preserve layer boundaries under pressure

The intended stack remains:

```text
TTIR
  -> standard VC4 value layer
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime
  -> hardware
```

Do not collapse layers to make a phase pass.

Examples:

- TTIR must not lower directly to `vc4kernel`.
- Value IR must not lower directly to `ssavc4` or scheduled `vc4`.
- Verified `vc4kernel` must not contain producer dialect operations.
- `vc4value` must remain tiny launch/policy plumbing.
- Default builds must not depend on optional Triton C++ dependencies.
- The optional Triton C++ dependency closure must remain quarantined behind the dedicated adapter target.

### E. Package quality bar

Every nontrivial phase package should have the following shape unless the user explicitly asks otherwise:

1. Read-only inventory/design lock.
2. Narrow implementation prompt(s) with explicit scope.
3. Static verifier/lit/audit coverage.
4. Hardware isolation fixtures for new executable semantics.
5. Mixed acceptance fixtures or mixed regression updates.
6. Claim/coverage/doc/audit lock.
7. Final acceptance that either unlocks the next phase or reports exact blockers.

Do not leave a phase with vague “future cleanup” items if they affect the foundation for the next phase. Either fix them in the phase or mark the phase blocked with a concrete blocker.

### F. Hardware and regression standard

If a phase changes executable semantics or any lowering used by executable semantics, it must run real hardware unless the prompt is explicitly read-only/static.

Hardware acceptance must preserve:

- real source input;
- all intermediate IR/artifacts;
- CPU oracle;
- sentinel checks;
- mismatch/launch-failure accounting;
- active QPU coverage appropriate to the feature, normally 12 active QPUs for mixed acceptance;
- nonzero output/work hash where relevant.

Do not reduce fixture coverage, simplify the kernel, or weaken expected results to make hardware pass.

### G. Debugging expectations

Debug aggressively and scientifically.

When a mixed fixture fails:

- preserve all artifacts;
- inspect every layer of the pipeline;
- run relevant isolation fixtures;
- remove/add features to isolate the interaction;
- instrument the current or lower layer if needed;
- consider lower-half bugs as possible;
- stop and flag true lower-half regressions rather than hiding them.

Do not merely stare at code or make speculative patches without testing the hypothesis.

### H. Copyable terminal commands

When writing commands meant to be pasted directly into a terminal, do not include shell comment lines beginning with `#`.

Explanations belong outside the command block. Copyable shell blocks should contain executable shell only.

### I. Failure is acceptable; false success is not

Use `FAILURE` or `BLOCKED` when:

- the long-term design is unclear;
- a correct implementation would require a larger prerequisite;
- a required command fails;
- a verifier/audit/hardware result fails;
- passing would require a shortcut;
- a source foundation issue is found during a later phase.

Final responses must name exact blockers and next actions. Do not claim success with hidden caveats that make the next phase unsafe.
