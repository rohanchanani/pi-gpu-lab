# m3-07-reductions-and-rotate: Reductions and rotate

    Goal: implement rotate/dataflow support and harden lowering enough for independent-vector reductions.

Implementation expectations:
- Add or finish pure `ssavc4.rotate`, `ssavc4.pack`, and `ssavc4.unpack` operations.
- Keep carrier types as `i32`, `f32`, `vector<16xi32>`, or `vector<16xf32>`; do not introduce i8/i16 storage value types.
- Immediate rotate amounts must be in [0, 15].
- Lower rotate through legal scheduled VC4 rotate paths and rely on scheduled verifiers for physical constraints.
- Harden no-spill liveness/register allocation for reduction dataflow pressure; reject unsupported pressure with a deterministic diagnostic rather than emitting unsafe code.
- Add `warp_reduce_sum_ssavc4` as an independent-vector fixture.

Forbidden work:
- Do not classify warp reductions as cooperative-block fixtures unless they actually require block-wide shared state.
- Do not implement full spilling or aggressive ADD/MUL bundling just to pass this slice.

Verification includes rotate/pack/unpack tests, conversion lit, M2-compatible artifacts, and hardware expected JSON.

    ## Report at the end of the slice

    Report the changed source files, the exact verifier command(s) run, whether hardware was required, and the first failing verifier packet if anything remains red. Do not include private reasoning. Do not claim success until the typed verifier entry for `m3-07-reductions-and-rotate` passes.
