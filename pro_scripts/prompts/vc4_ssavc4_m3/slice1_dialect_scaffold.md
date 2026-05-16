# m3-01-ssavc4-dialect-scaffold: Dialect scaffold

    Goal: create the separate `ssavc4` dialect skeleton and make `vc4-opt --show-dialects` report it.

Implementation expectations:
- Add `compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Base.td`, `SSAVC4Dialect.td`, `SSAVC4Dialect.h`, and matching CMake files.
- Add `compiler/lib/Dialect/SSAVC4/IR/SSAVC4Dialect.cpp` and CMake integration.
- Register `mlir::ssavc4::SSAVC4Dialect` in `vc4-opt` alongside `mlir::vc4::VC4Dialect`.
- Add only scaffold smoke tests, such as `compiler/test/Dialect/SSAVC4/show-dialects.mlir` and, if useful, a minimal empty-module roundtrip.
- Keep full `ninja -C compiler/build check-vc4` green after the scaffold lands.

Forbidden work:
- Do not add substantial op semantics yet.
- Do not add active lit tests for `ssavc4.load_imm`, `ssavc4.element_number`, `ssavc4.alu.*`, `ssavc4.make_flags`, type-model coverage, TMU/VPM/VDW/DMA, lowering, fixtures, or any later-slice feature.
- Do not check expected-red or future-slice tests into active lit paths early; add each test in the same slice that implements the corresponding op/type/lowering.
- Do not modify `VC4ArtifactEmitter.cpp`.
- Do not create structured `vc4` ops or `function_form<structured>`.

Verification commands include `ninja -C compiler/build vc4-opt`, `ninja -C compiler/build check-vc4`, `vc4-opt --show-dialects`, the SSAVC4 scaffold lit test, and the scaffold active-test hygiene scan.

    ## Report at the end of the slice

    Report the changed source files, the exact verifier command(s) run, whether hardware was required, and the first failing verifier packet if anything remains red. Do not include private reasoning. Do not claim success until the typed verifier entry for `m3-01-ssavc4-dialect-scaffold` passes.
