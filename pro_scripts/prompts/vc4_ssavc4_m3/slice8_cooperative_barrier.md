# m3-08-cooperative-barrier-basics: Cooperative barrier basics

    Goal: implement basic cooperative barrier semantics using scheduled semaphores and M2 resource metadata.

Implementation expectations:
- Add effectful `ssavc4.sema.acquire`, `ssavc4.sema.release`, and `ssavc4.barrier`.
- Require `vc4.resource` metadata that declares cooperative-block scheduling and barrier use.
- Validate `warps_per_block`, semaphore count, and full-block residency requirements.
- Lower to the known-good M2 four-semaphore barrier pattern where applicable.
- Preserve runtime logical block/warp identity from launch ABI uniforms; do not use physical QPU ID.
- Add `qpu_barrier_syncthreads_ssavc4` or a smaller barrier smoke that exercises the same cooperative resource contract.

Forbidden work:
- Do not use mutex as a substitute for cooperative block scheduling.
- Do not mutate M2 cooperative fixtures.
- Do not weaken resource metadata validation.

Verification includes sema/barrier tests, conversion tests, artifact resource contract checks, and hardware run.

    ## Report at the end of the slice

    Report the changed source files, the exact verifier command(s) run, whether hardware was required, and the first failing verifier packet if anything remains red. Do not include private reasoning. Do not claim success until the typed verifier entry for `m3-08-cooperative-barrier-basics` passes.
