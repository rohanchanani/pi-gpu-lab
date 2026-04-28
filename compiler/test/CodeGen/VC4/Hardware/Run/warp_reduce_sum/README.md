
warp_reduce_sum

warp_reduce_sum validates a single-QPU 16-lane floating-point reduction primitive.

This is the next incremental reduction test because reductions are foundational for ML and scientific kernels, but block-wide reductions first need a trusted per-QPU lane-reduction building block. The test is intentionally single-warp in semantics: each QPU loads one logical 16-lane vector, reduces the active lanes to one sum, replicates that sum across the vector, and stores the valid prefix through VPM/VDW.

The semantic operation is:

for each logical vector v:
  sum = Σ active_lanes input[v * 16 + lane]
  out[v * 16 + lane] = sum for active lanes

The final partial vector masks inactive lanes to zero before reduction and stores only the logical active prefix through dynamic VDW DEPTH.

The hardware path exercised here is:

TMU0 direct memory lookup
-> QPU vector ALU reduction by rotate/add sequence
-> replicated sum vector
-> mutex-protected VPM staging
-> dynamic-depth VDW DMA store

This test does not validate cross-QPU reductions, VPM shared memory, barriers, arbitrary segmented reductions, or integer reductions. It is only the per-QPU lane-reduction primitive.

Candidate-side codegen is intentionally disabled until the backend can emit the qasm/launcher bundle.

Do not add this test to compiler/test/CodeGen/VC4/catalog.json until a reference hardware run has passed and the result line has been validated against expected.json.

