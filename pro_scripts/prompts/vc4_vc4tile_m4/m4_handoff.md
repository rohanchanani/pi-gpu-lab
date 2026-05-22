# M4 Handoff

M4 begins from a repository where M3 has implemented `ssavc4` and `--convert-ssavc4-to-vc4`. The lower half already includes spill-aware allocation, spill-frame metadata, hidden spill uniforms, and block-argument / edge-copy lowering. M4 should treat these as existing lower-half capabilities.

M4 must create `vc4tile` as the next compiler layer above SSAVC4:

```text
future producer adapters -> vc4tile -> ssavc4 -> scheduled vc4 -> artifacts/runtime
```

The full `vc4tile` architecture should be defined early. Lowering and hardware proof are sliced vertically.

Important existing facts:

- VC4 has 12 QPU warp slots and 16 SIMD lanes.
- Independent-vector and cooperative-block schedule modes already exist in runtime/artifact metadata.
- Shared VPM uses one global 4 KiB user-visible VPM window.
- The block barrier uses a validated four-semaphore reusable protocol.
- Existing hardware fixtures prove SSAVC4 paths for vector store, SAXPY/TMU, block args, spills, reductions, shared VPM, and barrier.
