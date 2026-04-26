# Candidate side disabled

Candidate-side generation is intentionally disabled for `qpu_num_register`.

This test is exploratory hardware truth acquisition for the hardware
`QPU_NUMBER` register. The trusted reference qasm/c/h bundle is the source of
truth until the final-stage vc4 dialect and code generator can represent a
register-materialized `qpu_num` builtin directly.
