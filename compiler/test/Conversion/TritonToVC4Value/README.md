# TritonToVC4Value Conversion Tests

This suite is dedicated static coverage for the accepted Phase 7.5 C++ TTIR
frontend path:

```text
source-controlled TTIR
  -> vc4-triton-opt --convert-triton-to-vc4-value
  -> VC4 value IR
  -> vc4-opt --vc4-verify-value-surface
  -> vc4-opt --convert-vc4-value-to-vc4kernel --verify-vc4kernel
```

The default build remains Triton-free. Tests that require `vc4-triton-opt` are
guarded by `REQUIRES: vc4-has-triton-cpp-frontend` and are unsupported when the
optional frontend tool is absent.

Python semantic TTIR-to-value lowering is not used here. `READY_FOR_TRITON`
remains `NO`.
