# m3-04-global-store: Global store

    Goal: bring up the first observable hardware-style SSAVC4 path: vector/global store through VPM/VDW.

Implementation expectations:
- Add an effectful `ssavc4.vdw.store` or equivalent minimal target-specific VPM→VDW store op.
- `%addr` is scalar `i32`; `%value` is `vector<16xi32>` or `vector<16xf32>`; `elem_bytes = 4` in M3 v1; `active_lanes` is 1..16.
- Model required ordering with effects/tokens; do not make store pure.
- Lower using known-good M2 scheduled store patterns from `vector_store_smoke`, `global_store_coalesced_multi`, and `saxpy_full` generated examples.
- Add `compiler/test/CodeGen/SSAVC4/Support/run_ssavc4_candidate_codegen_test.sh`, which converts SSAVC4 input to scheduled VC4 and then reuses the M2 artifact runner.
- Add `vector_store_smoke_ssavc4` with SSAVC4 input, expected JSON, and candidate harness.

Forbidden work:
- Do not invent a new artifact format.
- Do not copy reference QASM into candidate outputs.
- Do not mutate M2 VC4 fixtures or generated examples.

Verification includes dialect/conversion lit tests, M2-compatible manifest/layout/QASM checks, candidate assemble/build, and hardware run.

    ## Report at the end of the slice

    Report the changed source files, the exact verifier command(s) run, whether hardware was required, and the first failing verifier packet if anything remains red. Do not include private reasoning. Do not claim success until the typed verifier entry for `m3-04-global-store` passes.
