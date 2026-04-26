# qpu_num_register

## Purpose

`qpu_num_register` is an exploratory hardware-ground-truth test for the VC4 QPU
hardware `QPU_NUMBER` register.

The reference kernel queues 16 user-program requests through the V3D QPU
scheduler path. Each requested QPU program:

1. reads the hardware `qpu_num` / `QPU_NUMBER` register,
2. writes one 32-bit observation to VPM,
3. stores that one word to its per-request output slot with VDW,
4. terminates with `thrend` plus the required two delay-slot instructions.

The harness prints one human-readable line for each requested program:

```text
Launch 1 QPU_NUM: <result>
Launch 2 QPU_NUM: <result>
...
Launch 16 QPU_NUM: <result>
```

It then prints the final oracle line:

```text
VC4_TEST_RESULT name=qpu_num_register status=PASS launches=16 completed_requests=16 invalid_results=0 pending_results=0 qpu_mask=0x...
```

## Why this is the right next step

The existing hardware-run corpus has already covered basic launch/completion,
output writes, simple input/output movement, TMU direct loads, and SAXPY-style
multi-QPU work distribution. This test advances one hardware/codegen concept:
**direct observation of the hardware QPU number selected by the scheduler**.

It is intentionally not a throughput or arithmetic test. It is a scheduler and
register-observation test.

## Hardware path exercised

- V3D user-program scheduler request queue.
- Hardware QPU register-map read of `qpu_num` / `QPU_NUMBER`.
- VPM write and VDW one-word DMA store for the observation.
- Program termination with `thrend` and two following delay-slot instructions.

The reference asks the scheduler for 16 user programs, even on systems expected
to have fewer physical QPUs. The goal is to observe which hardware QPU numbers
are assigned and whether QPUs are reused across the 16 requests.

## What this does not prove yet

- It does not prove a stable ordering of QPU assignment.
- It does not require a particular physical QPU count.
- It does not prove candidate/codegen support; `candidate/` is disabled.
- It does not prove a dialect lowering for a register-materialized `qpu_num`
  builtin. The current `input.mlir` is a verified metadata carrier for this
  exploratory hardware test because the current dialect may only model
  `qpu_num` as a uniform-suffix builtin.

## Semantic oracle

`expected.json` requires only stable facts:

- all 16 queued requests complete,
- all 16 output slots are written,
- every observed QPU number is in the 4-bit architectural range `[0, 15]`.

The observed sequence and bitmask are deliberately printed but not required.
They are hardware facts to inspect after the run, not a stable oracle yet.
