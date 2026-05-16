# m3-03-lowering-skeleton: Lowering skeleton

    Goal: add `--convert-ssavc4-to-vc4` and lower a minimal SSAVC4 kernel to scheduled `vc4`.

Implementation expectations:
- Add `compiler/include/vc4/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.h` and `compiler/lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp`.
- Register the pass in `vc4-opt` as `--convert-ssavc4-to-vc4`.
- Convert `ssavc4.module` to `vc4.module` and `ssavc4.func` to scheduled `vc4.func`.
- Copy `kernel`, `threading`, `vc4.launch_abi`, and `vc4.resource` metadata unchanged.
- Lower `ssavc4.thread_end` or the minimal kernel terminator to a conservative scheduled thread-end epilogue.
- Lower minimal `load_imm`/simple ALU if needed for the test.

Forbidden work:
- Do not attempt aggressive scheduling, useful delay-slot filling, TMU overlap, or spilling.
- Do not require `vc4-codegen` to parse SSAVC4 directly.

Verification commands include conversion lit tests and an SSAVC4 codegen lit test that pipes lowered scheduled output through the existing scheduled verifier pipeline.

    ## Report at the end of the slice

    Report the changed source files, the exact verifier command(s) run, whether hardware was required, and the first failing verifier packet if anything remains red. Do not include private reasoning. Do not claim success until the typed verifier entry for `m3-03-lowering-skeleton` passes.
