# m3-05-branch-tail-lowering: Branch and tail lowering

    Goal: add restricted flags/control flow and correct scheduled branch layout.

Implementation expectations:
- Add `ssavc4.make_flags`, `ssavc4.br`, and `ssavc4.cond_br` with strict `!ssavc4.flags` single-use restrictions.
- Support direct block successors and reject unsupported CFGs with deterministic diagnostics.
- Linearize and schedule non-branch operations first, assign final slot numbers, then insert `vc4.qpu.branch` ops.
- Fill every branch delay-slot region with exactly three scheduled QPU ops; M3 v1 may use nops only.
- Compute branch immediates after final scheduled layout. Do not hardcode guessed constants.
- Add a tail-control SSAVC4 fixture, `global_store_coalesced_multi_ssavc4`.

Forbidden work:
- Do not use the old `waddr_add = 39`, `waddr_mul = 39` assumption.
- Do not branch on fixture names.
- Do not hide branch failures by weakening the scheduled branch verifier.

Verification includes branch roundtrip/invalid tests, conversion lit with scheduled branch checks, QASM assembly, and hardware fixture verification.

    ## Report at the end of the slice

    Report the changed source files, the exact verifier command(s) run, whether hardware was required, and the first failing verifier packet if anything remains red. Do not include private reasoning. Do not claim success until the typed verifier entry for `m3-05-branch-tail-lowering` passes.
