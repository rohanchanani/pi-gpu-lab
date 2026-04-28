
block_reduce_sum

block_reduce_sum validates a CUDA-like block reduction across up to 12 QPU warps using VPM shared memory and the locked four-semaphore barrier.

This is the next incremental cooperative primitive after per-QPU reduction and reusable barrier validation. It combines per-warp partial sums, VPM shared-memory staging, gpu.barrier semantics, leader finalization, and a VPM/VDW store path in one real algorithm.

The semantic operation is:

for each block:
  sum = Σ input[block * values_per_block + i] for i in [0, values_per_block)
  out[block * 16 + lane] = sum for lane in [0, 16)

The reference implementation schedules one resident block wave at a time. This keeps barrier participation simple and correct: every logical warp in the block is resident before any warp reaches the four-semaphore barrier. The multi-block case is implemented as sequential resident waves, not simultaneous multi-resident blocks.

The hardware path exercised here is:

TMU0 direct memory lookup
-> per-warp scalar partial sum
-> VPM shared-memory partial rows
-> four-semaphore reusable barrier
-> leader reads partial rows from VPM
-> VPM/VDW store of the replicated block result

This test does not validate arbitrary reductions larger than one resident wave, atomics, global reductions across blocks, multi-resident block scheduling, or performance-tuned tree reductions. It also does not relax the VPM mutex rule.

Candidate-side codegen is intentionally disabled until the backend can emit the qasm/launcher bundle.

Do not add this test to compiler/test/CodeGen/VC4/catalog.json until a reference hardware run has passed and the result line has been validated against expected.json.

