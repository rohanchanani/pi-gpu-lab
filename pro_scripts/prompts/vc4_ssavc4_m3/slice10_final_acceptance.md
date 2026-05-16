# m3-10-final-acceptance: Final acceptance

    Goal: prove M3 is complete and M2 still passes.

Required final state:
- `ssavc4` parses, prints, verifies, and round-trips.
- Pure value ops are pure; hardware-state ops are effectful and/or token-ordered.
- `--convert-ssavc4-to-vc4` lowers supported SSAVC4 kernels to scheduled `vc4` only.
- Lowered scheduled output passes existing scheduled VC4 verifiers.
- SSAVC4-input fixtures generate M2-compatible artifacts and pass hardware expected JSON.
- No compiler source branches on fixture/public names.
- No normal lowering path uses physical QPU number for logical identity.
- The generic M2 verifier passes every M2 slice.

Do not add new features in final acceptance unless a verifier failure shows a narrow missing hardening step. Final acceptance should be mostly verification, documentation, and cleanup.

The final M2 regression command is the one in `pro_scripts/vc4_ssavc4_m3_verifications.json`. Do not weaken it or replace it with an M1 historical gate.

    ## Report at the end of the slice

    Report the changed source files, the exact verifier command(s) run, whether hardware was required, and the first failing verifier packet if anything remains red. Do not include private reasoning. Do not claim success until the typed verifier entry for `m3-10-final-acceptance` passes.
