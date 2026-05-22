# m3-03-lowering-skeleton: Lowering skeleton

    Goal: add `--convert-ssavc4-to-vc4` and lower a minimal SSAVC4 kernel to scheduled `vc4`.

Implementation expectations:
- Add `compiler/include/vc4/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.h` and `compiler/lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp`.
- Register the pass in `vc4-opt` as `--convert-ssavc4-to-vc4`.
- Convert `ssavc4.module` to `vc4.module` and `ssavc4.func` to scheduled `vc4.func`.
- Copy `kernel`, `threading`, `vc4.launch_abi`, and `vc4.resource` metadata unchanged.
- Lower `ssavc4.thread_end` or the minimal kernel terminator to a conservative scheduled thread-end epilogue.
- Lower minimal `load_imm`/simple ALU if needed for the test.
- Structure the lowering skeleton around explicit internal seams: instruction selection/templates, virtual values, liveness, allocation, conservative scheduling, hazard insertion, and branch layout. A simple implementation is fine, but it must not become a monolithic direct op-to-final-bundle converter that blocks future allocator/scheduler replacement.
- Historical slice note: this early slice originally allowed a trivial allocator that rejected excessive pressure. The current SSAVC4 lower half includes spill-frame support, so future work should preserve allocator/spill-planner seams rather than treating no-spill as the project state.

Forbidden work:
- Do not attempt aggressive scheduling, useful delay-slot filling, or TMU overlap in this early slice.
- Do not remove or bypass spill-frame support in current lower-half code; this slice text is historical context for the original scaffold.
- Do not require `vc4-codegen` to parse SSAVC4 directly.

Verification commands include conversion lit tests and an SSAVC4 codegen lit test that pipes lowered scheduled output through the existing scheduled verifier pipeline.

    ## Report at the end of the slice

    Report the changed source files, the exact verifier command(s) run, whether hardware was required, and the first failing verifier packet if anything remains red. Do not include private reasoning. Do not claim success until the typed verifier entry for `m3-03-lowering-skeleton` passes.

## Retry/source-product discipline

Every retry for this slice must preserve or recreate the exact required source products from the worklist and verifier:

- `compiler/include/vc4/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.h`
- `compiler/lib/Conversion/SSAVC4ToVC4/CMakeLists.txt`
- `compiler/lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp`
- `compiler/test/Conversion/SSAVC4ToVC4/minimal-thrend.mlir`
- `compiler/test/Conversion/SSAVC4ToVC4/metadata-copy.mlir`
- `compiler/test/CodeGen/SSAVC4/Emit/minimal-thrend-ssavc4.mlir`

If a later verifier/build/lit failure requires touching only one file, still keep the required conversion files and tests present. A patch that drops source products is not a valid repair.

## Lowering architecture policy

Historical slice note: M3 slice 3 began with a conservative allocator that could reject excessive pressure. The current lower half now has spill-frame support, so keep the allocator/spill-planner boundary explicit.

The implementation should keep visible internal seams for instruction selection/templates, virtual values, liveness, allocation, conservative scheduling, hazard insertion, and branch layout. Do not collapse the slice into a monolithic direct op-to-final-bundle converter that would block allocator/scheduler replacement.
