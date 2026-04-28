
vpm_setup_clobber candidate side

Candidate-side codegen is disabled for this hardware-run test.

The reference side is a low-level diagnostic litmus that uses physical QPU reservations, QPU_NUMBER dispatch, QPU semaphores, an intentionally unprotected VPM setup clobber window, and mutex-protected VPM/VDW readback. Candidate generation should remain disabled until the backend can faithfully represent and emit those operations.

Do not claim candidate/codegen coverage for this test until a generated qasm/launcher bundle is executed on hardware and checked against the same expected.json oracle.

