
Candidate side enabled

Candidate/codegen execution is enabled for conv1d_3tap.

The candidate harness launches the generated kernel on hardware and verifies it
against the same clamp-to-edge 3-tap host oracle used by the reference side.

Do not weaken the exact output comparison. The coefficients and input pattern
are binary-rational values chosen so the expected results are exactly
representable for this arithmetic order.
