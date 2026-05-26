# matmul_blocked

`matmul_blocked` is the shared-memory/tiled successor to `matmul_naive`.

It computes row-major:

```text
C[M,N] = A[M,K] * B[K,N]
```

The source product is `input.mlir`, a final-stage scheduled VC4 dialect kernel.
The candidate harness launches that generated kernel on the real device and
compares against an unweakened CPU oracle with exact f32 bit checks.

## Kernel Shape

- one QPU request is one 16-lane logical warp;
- one cooperative block has 12 logical warps;
- one resident block computes one `12 x 16` output tile;
- `logical_warp_id` selects the output row within the tile;
- `ELEMENT_NUMBER` selects the output column lane;
- the block stages a `12 x 16` B tile in shared VPM rows;
- all warps execute a four-semaphore reusable barrier after B-tile loads and
  another barrier after VPM reads;
- global loads use TMU direct memory lookup;
- global stores use VPM staging plus VDW DMA stores;
- VPM setup/access and VDW setup/store sequences are protected by the global
  QPU mutex.

## Verification

The harness sweeps ten cases, including zero-row/zero-column, `K = 0`, exact
tile shape, tail tiles, and larger dynamic shapes up to `193 x 217 x 97`.

For every case it:

- packs A/B/C into padded device scratch;
- launches one generated cooperative tile-wave per output tile;
- unpacks live C elements;
- compares every live output bit-for-bit with the CPU matmul oracle;
- checks compact output guards, real-row padded C tails, and the device C guard.

The current hardware result requires 362 cooperative tile-wave launches, zero
mismatches, zero sentinel mismatches, and `max_abs_diff=0.0`.
