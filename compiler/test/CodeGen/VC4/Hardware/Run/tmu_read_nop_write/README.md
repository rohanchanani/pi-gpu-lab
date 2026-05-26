# VC4 hardware test: `tmu_read_nop_write`

This is the TMU-backed counterpart to `read_nop_write`.

## What this test verifies

`read_nop_write` already proved the VDR DMA-load path:

```text
host input -> VDR DMA load -> VPM -> QPU register -> VPM -> VDW DMA store -> host output
```

This test keeps the same output path and semantic oracle, but replaces the input
path with the TMU direct-memory lookup path:

```text
host input -> TMU0 direct memory lookup -> r4 -> VPM -> VDW DMA store -> host output
```

Each active QPU copies exactly one 16-lane vector of `u32` values. The public
launcher API exposes only semantic host-side buffers and a word count. The
physical per-QPU uniforms are packed internally by the launcher:

```text
[0] input base address
[1] result base address
[2] word count
[3] qpu_id
[4] num_qpus
```

The kernel computes:

```text
base_word = qpu_id * 16
for lane in 0..15:
  result[base_word + lane] = input[base_word + lane]
```

The reference qasm issues one vector of direct TMU0 memory requests by writing
absolute byte addresses to `t0s`, then receives the vector through `ldtmu0`
into `r4`. The received vector is written unchanged into one VPM row and stored
back to memory through VDW.

## Why this is hardware-relevant

The VideoCore IV TMU supports general 32-bit direct-address memory lookups by
writing only the `s` parameter; such lookups do not consume texture-setup
uniforms and ignore the bottom two address bits. The result is received through
the TMU receive path into `r4`.

This test intentionally exercises only one outstanding direct request vector per
QPU. It is not a TMU FIFO-depth, TMU pipelining, texture descriptor, or
thread-switching test.

## What this test does not prove

- It does not prove TMU texture-mode descriptor lowering.
- It does not prove multiple outstanding TMU requests.
- It does not prove TMU0/TMU1 load balancing.
- It does not prove tail-safe handling.

## Expected result

For the default bare-metal runtime, `active_qpus = 12`, so the test copies
`12 * 16 = 192` `u32` words. The input pattern is `input[i] = i + 1`, so the
expected checksum is:

```text
1 + 2 + ... + 192 = 18528
```

The candidate side builds the generated bundle from `input.mlir`, launches it on
hardware, compares all 192 copied words with an exact host oracle, verifies an
output guard region, and checks the runtime launch counters. The final
successful result line is:

```text
VC4_TEST_RESULT name=tmu_read_nop_write status=PASS checked_elements=192 mismatches=0 sentinel_mismatches=0 launch_failures=0 active_qpus=12 lanes=16 words=192 checksum=18528 runtime_allocations=1 runtime_launches=1 elapsed_usec=<informational>
```
