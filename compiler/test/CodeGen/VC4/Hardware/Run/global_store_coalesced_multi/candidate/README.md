
global_store_coalesced_multi candidate side

Candidate-side codegen is disabled for this hardware-run test.

The reference side is the hardware source of truth for tail-safe coalesced VPM/VDW global stores from multiple QPUs. Candidate generation should remain disabled until the backend can emit the qasm/launcher bundle and run it against the same expected.json oracle.

Do not claim candidate/codegen coverage for this test until a generated bundle has passed on hardware.

