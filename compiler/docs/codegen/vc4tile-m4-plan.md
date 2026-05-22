# VC4 M4 Plan: VC4Tile Dialect and Lowering to SSAVC4

## 1. Goal

M4 implements the VC4 Tile dialect (`vc4tile`) and lowers `vc4tile` to `ssavc4`. It is accepted only when VC4Tile-authored fixtures lower through the existing lower half and pass hardware/reference checks where executable.

```text
vc4tile
  ↓ --convert-vc4tile-to-ssavc4
ssavc4
  ↓ --convert-ssavc4-to-vc4
scheduled vc4
  ↓ vc4-codegen --emit-bundle
QASM / shader arrays / kernel_launch.c/h / manifest/layout
  ↓ vc4_runtime
hardware
```

Producer integrations are deliberately after M4:

```text
Triton TTIR -> vc4tile
IREE late executable/codegen IR -> vc4tile
IREE HAL runtime wrapper around vc4_runtime
JAX/PyTorch through IREE -> VC4
```

## 2. Verification philosophy

The milestone uses canonical feature contracts. Every implemented feature must be verified at the dialect boundary, the invalid/diagnostic boundary, the `vc4tile -> ssavc4` lowering boundary, the scheduled/artifact/runtime-metadata boundary, and hardware/reference boundary when executable.

Hardware is the gold standard. If a feature can run on hardware, M4 should run it on hardware and compare against a CPU/reference oracle or fixture-provided expected JSON that was produced from a CPU/reference check.

## 3. Slices

### m4-00: milestone package

Installs the descriptor, worklist, verification spec, context profiles, prompt package, and M4 docs. No compiler source changes.

### m4-01: dialect scaffold

Adds the `vc4tile` dialect skeleton, CMake/TableGen integration, `vc4-opt` registration, and dialect visibility smoke.

### m4-02: full op/type/attr contract

Defines the broad `vc4tile` semantic surface: kernel, identity, lane range, masks, masked global memory, rotate/reduce, shared VPM, barrier, schedule/resource attributes, traits, effects, roundtrip tests, and invalid tests. Later slices lower the features.

### m4-03: lowering skeleton and metadata

Adds `--convert-vc4tile-to-ssavc4`, lowers a minimal kernel to `ssavc4.func`, generates/carries `vc4.launch_abi` and `vc4.resource`, and installs the VC4Tile candidate support runner.

### m4-04: independent tile IDs, arithmetic, masks

Implements `program_id`, lane range, `ELEMENT_NUMBER`-derived lanes, splats, arithmetic, comparisons, and tail masks.

### m4-05: masked coalesced global store

Lowers `vc4tile.masked_store_global` through SSAVC4 VDW for coalesced contiguous stores. This is the first VC4Tile hardware slice.

### m4-06: masked global load and SAXPY

Lowers masked global loads through SSAVC4 TMU request/read and proves a SAXPY/vector-add style hardware fixture.

### m4-07: control flow, tail, block args

Lowers uniform branches, simple loops/merges, and tail masks using SSAVC4 successor operands/block arguments. It relies on the current spill-aware lower half rather than avoiding CFG.

### m4-08: rotate/reduce

Lowers warp-local rotate/reduce forms through SSAVC4 rotate/reduction support and proves a warp reduction fixture.

### m4-09: cooperative block resources

Adds cooperative schedule mode, logical block/warp/thread identity, resource metadata, and hardware proof that logical identity is runtime-supplied, not physical QPU-derived.

### m4-10: shared VPM

Lowers shared tile allocation/load/store to SSAVC4 VPM operations and resource metadata, then proves shared tile behavior on hardware.

### m4-11: barrier

Lowers `vc4tile.barrier` to SSAVC4 barrier/semaphore operations using full-residency cooperative metadata and the four-semaphore reusable barrier model.

### m4-12: final acceptance

Runs cumulative M4 contracts, all active VC4Tile hardware fixtures, full `check-vc4`, M3/M2 lower-half regression, and a read-only Codex implementation-integrity audit.

## 4. Pitfalls this package must prevent

- Redefining M4 as `gpu -> ssavc4`.
- Lowering `vc4tile` directly to scheduled `vc4`.
- Making `vc4tile` so low-level that it exposes TMU/VDW setup words, raw semaphore protocols, physical registers, hazards, or QASM.
- Making `vc4tile` so high-level that it accepts tensors/linalg/StableHLO/Torch/Triton/IREE IR directly.
- Treating VPM as arbitrary CUDA shared SRAM.
- Treating VDW as arbitrary scalar scatter store support.
- Permitting divergent barrier participation.
- Using physical `QPU_NUMBER` for logical identity.
- Passing fixtures with hard-coded names, fake logs, reference QASM, or expected-result injection.
- Assuming the old no-spill M3 plan is still current; the current lower half has spill and block-argument support.

## 5. Final acceptance criteria

M4 is accepted only when:

- `vc4tile` parses, prints, verifies, and roundtrips.
- Unsupported inputs fail with deterministic diagnostics.
- `--convert-vc4tile-to-ssavc4` lowers supported features to intended SSAVC4 forms.
- Lowered SSAVC4 continues through scheduled VC4 verifiers and artifact emission.
- VC4Tile hardware fixtures pass CPU/reference comparisons.
- Resource metadata matches runtime limits and semantics.
- No direct producer-lowering or direct-to-VC4 shortcuts exist.
- A read-only Codex integrity audit returns `integrity_pass = YES`.
- Full lower-half regression remains green.
