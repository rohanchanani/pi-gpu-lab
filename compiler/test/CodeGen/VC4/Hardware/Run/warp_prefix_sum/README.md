
warp_prefix_sum

warp_prefix_sum validates a single-QPU 16-lane inclusive prefix-sum primitive with exact unsigned 32-bit integer results and tail masking.

This is the next incremental warp-local primitive after reduction because prefix scan stresses lane-relative data movement and staged accumulation across a SIMD vector. It remains smaller and more deterministic than floating-point softmax or layernorm because the reference uses small unsigned integer inputs and exact CPU oracles.

The semantic operation is:

for each logical vector v:
  out[v * 16 + lane] = input[v * 16 + 0] + ... + input[v * 16 + lane]

Only active lanes are copied back for the final partial vector. The private input scratch is padded to a 16-element boundary with zeroes, and the VDW store uses dynamic DEPTH so the host-visible guard region past n stays 0xdeadbeef.

The hardware path exercised here is:

TMU0 direct memory lookup
-> QPU integer vector rotate/mask/add inclusive scan
-> mutex-protected VPM staging
-> dynamic-depth VDW DMA store

This test does not validate block-wide prefix scan, stream compaction, segmented scan, floating-point scans, shared memory, or barriers.

Candidate-side codegen is intentionally disabled until the backend can emit the qasm/launcher bundle.

Do not add this test to compiler/test/CodeGen/VC4/catalog.json until a reference hardware run has passed and the result line has been validated against expected.json.

