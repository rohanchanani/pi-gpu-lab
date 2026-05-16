# m3-02-type-model-and-pure-ops: Type model and pure ops

    Goal: define the lowerable SSAVC4 value type model, minimal custom types/attrs, and pure value operations.

Implementation expectations:
- Use builtin `i32`, `f32`, `vector<16xi32>`, and `vector<16xf32>` for ordinary data.
- Add only minimal custom types: `!ssavc4.async.token`, `!ssavc4.tmu.desc`, `!ssavc4.vpm.desc`, and `!ssavc4.flags` as needed by the planned operation set.
- Add helper predicates in C++ for scalar32/vector16/int-carrier/float-carrier/value-type shape checks.
- Add pure ops: `load_imm`, `element_number`, `splat`, `mov`, `alu.add`, `alu.mul`, and, if straightforward, `pack`, `unpack`, `rotate`, and `make_flags` with single-use restrictions.
- Reuse only live scheduled-VC4 attrs/enums. Define SSAVC4 attrs for TMU/VPM/VDW/flags descriptor concepts that no longer exist in `vc4`.

Forbidden work:
- Do not mark uniform/TMU/VPM/VDW/semaphore/barrier/thread_end ops pure.
- Do not introduce `!ssavc4.v16f32` or similar custom ordinary data vector types.
- Do not expose physical register addresses in user-facing SSAVC4 syntax.

Verification commands include dialect roundtrip tests and invalid verifier tests for illegal types, shapes, and flags usage.

    ## Report at the end of the slice

    Report the changed source files, the exact verifier command(s) run, whether hardware was required, and the first failing verifier packet if anything remains red. Do not include private reasoning. Do not claim success until the typed verifier entry for `m3-02-type-model-and-pure-ops` passes.
