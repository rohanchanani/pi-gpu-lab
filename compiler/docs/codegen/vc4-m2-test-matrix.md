# VC4 M2 Test Matrix

M2 organizes hardware verification around fixture matrices consumed by the `fixture_matrix` verifier mechanism.

Required matrix groups:

- `smoke_runtime`: minimal program/runtime smoke tests.
- `independent_vector`: independent QPU request kernels.
- `memory_sfu`: TMU/SFU/VPM/VDW memory path coverage.
- `cooperative_shared_reduction`: barrier/shared/reduction coverage.
- `m2_required`: final required union.

Fixtures may be added or adapted in the dedicated M2 test-generation stage. Existing reference/oracle files remain immutable unless a slice explicitly adds a new fixture.
