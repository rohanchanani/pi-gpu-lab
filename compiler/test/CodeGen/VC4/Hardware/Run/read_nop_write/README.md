# VC4 hardware ground-truth test: `read_nop_write`

This test is the next incremental hardware contract test after
`memory_output`.

## Purpose

`read_nop_write` proves the simplest full memory round trip through the QPU
general-purpose path:

1. the host launcher copies a deterministic `uint32_t` input vector into
   device-visible memory;
2. each active QPU receives a private uniform stream;
3. each QPU DMA-loads one 16-word row from main memory into VPM using VDR;
4. the QPU reads that row from VPM into a vector register;
5. the QPU writes the unchanged vector back to VPM;
6. the QPU DMA-stores that row from VPM back to main memory using VDW;
7. the host copies the output back and checks it against the input.

This is intentionally a no-op copy kernel. It is stronger than
`memory_output`, because it exercises both the input and output sides of the
VPM/VDR/VDW path, but it still avoids arithmetic, loops, tails, TMU, SFU, and
thread switching.

## What it verifies

- V3D/QPU user-program launch still works.
- Per-QPU uniform packing works for a buffer input, buffer output, scalar `n`,
  `logical_request`, and `total_requests`.
- VDR can DMA one 16-lane `u32` vector per active QPU from memory into VPM.
- A QPU can read the vector back from VPM.
- A QPU can write an unchanged vector into VPM.
- VDW can DMA the vector back to host-visible memory.
- The host can verify the copied words deterministically.

## What it does not verify

- It does not verify arithmetic.
- It does not verify loops or non-one-vector-per-QPU work distribution.
- It does not verify tail-safe handling.
- It does not verify TMU asynchronous loads.
- It does not verify threadable kernels.

## Public candidate API

```c
int read_nop_write_launch(
    struct vc4_runtime *rt,
    const uint32_t *input,
    uint32_t *result,
    uint32_t n);
```

The public API exposes semantic arguments only. It does not expose `qpu_id`,
`num_qpus`, raw uniform arrays, VPM rows, V3D scheduler registers, or GPU bus
addresses.

## Candidate physical uniform stream

For each active QPU, the generated launcher packs:

```text
[0] input base address
[1] result/output base address
[2] n word count
[3] logical_request
[4] total_requests
```

The kernel uses:

```text
byte_offset = logical_request * 16 * sizeof(uint32_t)
input_addr  = input_base  + byte_offset
result_addr = result_base + byte_offset
vpm_row     = logical_request
```

This test's launcher requires `n == active_qpus * 16`. That is a test-specific
policy, not a global VC4 backend tail policy.

## Expected result

The harness fills `input[i] = i + 1` for `active_qpus * 16` words. With the
current reference runtime policy of 12 active QPUs, this is 192 words and the
expected checksum is:

```text
1 + 2 + ... + 192 = 18528
```

The final successful line is:

```text
VC4_TEST_RESULT name=read_nop_write status=PASS checked_elements=192 mismatches=0 sentinel_mismatches=0 launch_failures=0 active_qpus=12 lanes=16 words=192 checksum=18528 runtime_allocations=1 runtime_launches=1 elapsed_usec=<informational>
```

`elapsed_usec` is informational and is not checked by `expected.json`.
