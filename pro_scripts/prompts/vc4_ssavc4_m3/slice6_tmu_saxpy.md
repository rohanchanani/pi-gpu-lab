# m3-06-tmu-direct-load-saxpy: TMU direct load and saxpy

    Goal: implement direct TMU load sufficient for `saxpy_full_ssavc4`.

Implementation expectations:
- Add effectful `ssavc4.tmu.request` and `ssavc4.tmu.read` with explicit token ordering.
- Support `tmu0`, `direct`, and `raw32` first. Defer TMU1, texture/cubemap modes, packed parts, and overlap optimizations unless explicitly needed.
- Direct-mode address operands may be scalar `i32` or `vector<16xi32>`.
- Lower requests to scheduled TMU address/parameter writes and reads to the scheduled receive sequence.
- Preserve `r4` lifetime after TMU receive.
- Use conservative non-overlapped scheduling; correctness is more important than throughput in M3 v1.
- Add `saxpy_full_ssavc4` with SSAVC4 input and the existing M2 artifact/harness semantics.

Forbidden work:
- Do not infer host copies from launch ABI metadata.
- Do not use physical QPU number for logical identity.
- Do not special-case `saxpy_full` strings in lowering or emitter code.

Verification includes TMU dialect/invalid tests, conversion tests, M2-compatible artifacts, and hardware expected JSON.

    ## Report at the end of the slice

    Report the changed source files, the exact verifier command(s) run, whether hardware was required, and the first failing verifier packet if anything remains red. Do not include private reasoning. Do not claim success until the typed verifier entry for `m3-06-tmu-direct-load-saxpy` passes.
