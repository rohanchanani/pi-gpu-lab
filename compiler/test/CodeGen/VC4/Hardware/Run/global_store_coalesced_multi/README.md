
global_store_coalesced_multi

global_store_coalesced_multi validates tail-safe coalesced global stores from multiple VC4 QPUs through the VPM/VDW store path.

This is the right next incremental store-path regression because real VC4 kernels ultimately need VPM/VDW for global writes: the QPU has no CUDA-like scalar global store instruction. Existing SAXPY and matmul tests exercise this path while also exercising TMU loads and arithmetic. This test isolates the generated-store abstraction: vector registers are staged into VPM rows, DMA-stored through VDW with dynamic tail depth, and distributed across many QPUs writing disjoint contiguous regions.

The semantic operation is:

out[i] = 0x51000000 | ((case_id & 0xff) << 16) | (i & 0xffff)

for every i in [0, n). Guard words beyond n must remain QPU_STORE_SENTINEL.

The hardware path exercised here is:

QPU vector register value construction
-> mutex-protected VPM row staging
-> dynamic-depth VDW DMA store
-> repeated launches using one persistent GPU allocation/code copy

This test does not validate TMU loads, floating-point arithmetic, barriers, shared memory, scatter stores, atomics, or VPM setup sharing. It only validates coalesced affine global stores through the conservative VPM/VDW path.

Candidate-side codegen is intentionally disabled until the backend can emit the qasm/launcher bundle.

Do not add this test to compiler/test/CodeGen/VC4/catalog.json until a reference hardware run has passed and the result line has been validated against expected.json.

