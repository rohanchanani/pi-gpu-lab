# m3-09-shared-vpm-cooperative: Shared VPM cooperative path

    Goal: implement the minimal shared VPM behavior needed for a cooperative shared-memory-style fixture.

Implementation expectations:
- Add effectful `ssavc4.vpm.read` and `ssavc4.vpm.write` only for the subset needed by the fixture.
- Keep descriptor modeling minimal and SSAVC4-owned; do not recreate removed structured `vc4.vpm.*` operations.
- Preserve cooperative-block metadata such as shared VPM rows/bytes, barrier use, full-block residency, and semaphores per block.
- Lower to scheduled VPM setup/read/write sequences consistent with M2 `block_reduce_sum` and `shared_transpose_16x16` evidence.
- Add `shared_transpose_16x16_ssavc4`, or explicitly reduce scope inside the slice while keeping a real shared-VPM cooperative fixture.

Forbidden work:
- Do not treat `vpm_setup_clobber` as an acceptance fixture; it is hardware characterization.
- Do not use broad generic DMA descriptors unless the fixture truly requires them.

Verification includes shared VPM tests, conversion tests, artifact checks, and hardware expected JSON.

    ## Report at the end of the slice

    Report the changed source files, the exact verifier command(s) run, whether hardware was required, and the first failing verifier packet if anything remains red. Do not include private reasoning. Do not claim success until the typed verifier entry for `m3-09-shared-vpm-cooperative` passes.
