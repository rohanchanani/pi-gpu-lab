
tmu_strided_load

tmu_strided_load validates direct TMU global loads for coalesced, strided, offset, and tail-masked read patterns.

This is the load-side counterpart to the coalesced global-store test. The default VC4 policy for ordinary global memory loads is TMU direct memory lookup, but existing SAXPY and matmul tests mix TMU loads with richer arithmetic and kernel structure. This test isolates address formation, stride handling, offsets, tail-lane padding, and repeated launches under the one-allocation runtime discipline.

The semantic operation is:

out[i] = scale * input[offset + i * stride] + bias

for every i in [0, n). The launcher pads private GPU input scratch so inactive tail-lane TMU reads stay inside allocated memory. The output guard region past n must remain sentinel.

The hardware path exercised here is:

QPU vector address formation
-> TMU0 direct memory lookup
-> r4 result consumption
-> fmul/fadd
-> mutex-protected VPM staging
-> dynamic-depth VDW DMA store

This test does not validate texture-format sampling, filtering, shared memory, barriers, global scatter stores, or VPM setup sharing. It uses TMU only for direct 32-bit memory lookup.

Candidate-side codegen is intentionally disabled until the backend can emit the qasm/launcher bundle.

Do not add this test to compiler/test/CodeGen/VC4/catalog.json until a reference hardware run has passed and the result line has been validated against expected.json.

