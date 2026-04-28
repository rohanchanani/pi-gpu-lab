# matmul_blocked

`matmul_blocked` is the shared-memory/tiled successor to `matmul_naive`.

It computes row-major:

```text
C[M,N] = A[M,K] * B[K,N]
```

The reference qasm is intentionally CUDA-like while staying faithful to the VC4
hardware path:

- one QPU request is one 16-lane logical warp;
- one cooperative block has 12 logical warps;
- one resident block computes a `12 x 16` output tile;
- `logical_warp_id` selects the output row within the tile;
- `ELEMENT_NUMBER` selects the output column lane;
- the block stages a `12 x 16` B tile in VPM rows `0..11`;
- all warps execute a four-semaphore reusable barrier after B-tile loads and
  another barrier after reads, matching the locked `__syncthreads()` protocol;
- global loads use TMU direct memory lookup;
- global stores use VPM staging plus VDW DMA stores;
- VPM setup/access and VDW setup/store sequences are protected by the global
  QPU mutex.

The launcher follows the current runtime discipline for multi-case tests:

```text
prepare once: allocate GPU memory, lock it, copy code once
launch many:  overwrite payload/uniforms and dispatch already-resident code
```

The bare-metal harness intentionally leaves the one allocation live until the
hardware runner power-cycles the Pi. This avoids repeated
`alloc/lock/unlock/free` cycles while testing multiple matrix shapes.

## Cases

The reference harness runs eight semantic launch cases covering:

- zero-row and zero-column no-output cases;
- `K = 0` zero-accumulation output;
- 1x1 scalar behavior;
- a small in-tile case;
- exact `12 x 16 x 12` tile shape;
- M/N/K tails;
- a larger `25 x 31 x 25` shape requiring multiple output tiles and multiple K
  tiles.

The stable oracle requires zero data mismatches, zero host sentinel-tail
mismatches, exactly one runtime allocation, eight semantic launches, thirteen
resident tile-wave dispatches, no scheduler timeouts, and no relevant
`V3D_ERRSTAT` changes.

## What this proves

This test checks a first CUDA-shaped blocked matmul lowering:

```text
TMU global loads -> VPM shared B tile -> semaphore barrier -> QPU fmul/fadd -> VPM/VDW global store
```

It specifically exercises VPM as CUDA-like shared memory and the reusable
barrier in an actual compute kernel rather than in a standalone synchronization
litmus.

## What this does not prove yet

This is not an optimized GEMM. It does not prove arbitrary shared-memory
scatter/gather, register blocking, coalesced multi-row stores, global memory
fence semantics, or VPM setup-state privacy. The implementation deliberately
keeps the conservative global mutex around VPM/VDW setup and access.
