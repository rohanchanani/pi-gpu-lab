# candidate side disabled

Candidate-side generation is intentionally disabled for `qpu_barrier_syncthreads`
until the VC4 code generator can emit a deployable qasm + launcher bundle for
register-materialized QPU builtins, VPM/VDW I/O, mutexes, and QPU semaphore
barriers.

The trusted reference side is the hardware source of truth for this exploratory
barrier-lowering test.
