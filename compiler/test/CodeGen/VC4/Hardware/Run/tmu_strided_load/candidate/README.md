
tmu_strided_load candidate side

Candidate-side codegen is disabled for this hardware-run test.

The reference side is the hardware source of truth for TMU0 direct-memory loads with coalesced, strided, offset, and tail-masked read patterns. Candidate generation should remain disabled until the backend can emit the qasm/launcher bundle and run it against the same expected.json oracle.

Do not claim candidate/codegen coverage for this test until a generated bundle has passed on hardware.

