# qpu_barrier_syncthreads

## Purpose

`qpu_barrier_syncthreads` validates a four-semaphore reusable QPU barrier as
the initial hardware-grounded lowering target for CUDA `__syncthreads()` and
MLIR `gpu.barrier` on VC4 user QPU programs.

The test asks whether all logical QPU warps in a fully resident cooperative
block can:

1. write distinct lane-tagged rows in the global user-visible VPM window,
2. execute a reusable semaphore barrier,
3. read all peer rows, and
4. observe only the current barrier generation's writes.

A second barrier after the read phase prevents a fast warp from overwriting its
row for the next iteration while a slower warp is still reading the current
iteration.

## Scope implemented by this test

This is the requested first correctness bundle for the barrier protocol.  It
implements five modes:

| mode | name | blocks | warps/block | iterations | QPU reservation mask |
|---:|---|---:|---:|---:|---:|
| 0 | same_slice_smoke | 1 | 2 | 4 | `{0,1}` |
| 1 | cross_slice_0_1 | 1 | 2 | 4 | `{0,QUPS}` |
| 2 | cross_slice_0_2 | 1 | 2 | 4 | `{0,2*QUPS}` |
| 3 | full_block_stress | 1 | 12 | 64 | `{0..11}` |
| 4 | two_block_partition | 2 | 4 | 32 | `{0..7}` |

Mode 4 uses disjoint VPM row ranges and disjoint semaphore groups for the two
resident logical blocks.

This test does not measure barrier performance and does not answer whether VPM
setup state is private, per-slice shared, or globally shared.  VPM setup/access
and VDW result stores remain protected by the global QPU mutex.

## Hardware path exercised

The reference bundle runs general-purpose QPU user programs through the V3D user
program scheduler.  The host reads `V3D_IDENT1`, writes `V3D_VPMBASE = 16` to
reserve the 4 KiB user-visible VPM window, clears scheduler/cache state for each
mode, and uses `V3D_SQRSV0/1` to reserve all QPUs except the selected physical
coverage set for the mode.

The QPU program reads `QPU_NUMBER` and `ELEMENT_NUMBER`, writes and reads
horizontal 32-bit VPM rows, uses the global QPU mutex around VPM/VDW setup and
access, and uses QPU semaphore instructions for the reusable barrier protocol.

## Barrier protocol

For `N = warps_per_block`, each barrier uses four system-wide 4-bit counting
semaphores:

```text
arrive, release, depart, reset
```

Non-leader logical warps run:

```text
sem_inc(arrive)
sem_dec(release)
sem_inc(depart)
sem_dec(reset)
```

Logical warp 0 is the leader and runs:

```text
repeat N-1: sem_dec(arrive)
repeat N-1: sem_inc(release)
repeat N-1: sem_dec(depart)
repeat N-1: sem_inc(reset)
```

After a complete barrier, all four semaphore counts should be back at zero.
The QPU semaphore instruction encodes the semaphore number as an immediate, so
this reference qasm statically dispatches by `block_id`: block 0 uses semaphores
`0..3`, and block 1 uses semaphores `4..7`.  The uniform stream still carries the
semantic semaphore IDs to document the future launcher/codegen ABI.

## Tag format

Every logical warp writes one lane-distinctive vector per iteration:

```text
tag(run_id, block_id, iter, logical_warp_id, lane) =
  0xb0000000 |
  ((run_id          & 0x0f) << 24) |
  ((block_id        & 0x0f) << 20) |
  ((iter            & 0xff) << 12) |
  ((logical_warp_id & 0x0f) <<  8) |
  (lane             & 0x0f)
```

A peer row is considered seen only when all 16 SIMD lanes match the expected
current-generation tag.

## Result model

Each QPU request stores compact per-warp vectors to host memory:

- physical QPU number reported by all lanes,
- `all_seen_mask_by_lane[16]`,
- `always_seen_mask_by_lane[16]`,
- `mismatch_count_by_lane[16]`.

The harness summarizes those vectors into the required `BARRIER_WARP` lines and
validates that every lane has:

```text
all_seen_mask_by_lane    == (1 << warps_per_block) - 1
always_seen_mask_by_lane == (1 << warps_per_block) - 1
mismatch_count_by_lane   == 0
```

## Output

Bare-metal execution cannot create host filesystem files directly.  Instead the
harness prints file-like sections into `reference/run.log`:

```text
vc4_barrier_topology.json:
vc4_barrier_runs.csv:
BARRIER_WARP ...
```

The final semantic oracle line is:

```text
VC4_TEST_RESULT name=qpu_barrier_syncthreads status=PASS runs=5 same_slice_pass=1 cross_slice_0_1_pass=1 cross_slice_0_2_pass=1 full_block_pass=1 multi_block_pass=1 qpu_mismatches=0 data_mismatches=0 timeouts=0 invalid_topology=0 errstat_relevant_changed=0 ...
```

`expected.json` intentionally ignores elapsed time and requires only stable
correctness fields.

## Interpretation

If this test passes, the four-semaphore reusable QPU barrier has demonstrated
the required ordering property for same-slice two-warp blocks, cross-slice
two-warp blocks, one full 12-QPU block over 64 iterations, and two resident
four-warp blocks with disjoint VPM rows and disjoint semaphore IDs.

That supports using this protocol as the initial VC4 backend implementation of
`gpu.barrier` / `__syncthreads()` for fully resident cooperative blocks, subject
to the runtime invariants documented in the test specification: block warps must
fit in one resident wave, barrier-using resident blocks must use disjoint VPM row
ranges and semaphore IDs, and VPM/VDW setup remains mutex-protected until a
separate setup-clobber test says otherwise.
