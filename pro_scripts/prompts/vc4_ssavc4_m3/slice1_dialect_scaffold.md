# m3-01-ssavc4-dialect-scaffold: Dialect scaffold

    Goal: create the separate `ssavc4` dialect skeleton and make `vc4-opt --show-dialects` report it.

Implementation expectations:
- Add `compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Base.td`, `SSAVC4Dialect.td`, `SSAVC4Dialect.h`, and matching CMake files.
- Add `compiler/lib/Dialect/SSAVC4/IR/SSAVC4Dialect.cpp` and CMake integration.
- Register `mlir::ssavc4::SSAVC4Dialect` in `vc4-opt` alongside `mlir::vc4::VC4Dialect`.
- Add `compiler/test/Dialect/SSAVC4/show-dialects.mlir`.

Forbidden work:
- Do not add substantial op semantics yet.
- Do not modify `VC4ArtifactEmitter.cpp`.
- Do not create structured `vc4` ops or `function_form<structured>`.

Verification commands include `ninja -C compiler/build vc4-opt`, `vc4-opt --show-dialects`, and the SSAVC4 scaffold lit test.

    ## Report at the end of the slice

    Report the changed source files, the exact verifier command(s) run, whether hardware was required, and the first failing verifier packet if anything remains red. Do not include private reasoning. Do not claim success until the typed verifier entry for `m3-01-ssavc4-dialect-scaffold` passes.
