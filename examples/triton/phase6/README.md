# Phase 6 Triton TTIR Corpus

This directory contains the Phase 6 real Triton source corpus and source-controlled TTIR snapshots generated from pinned Triton 3.7.0.

The required Phase 7 elementwise candidates are generated and parse-checked:

- `kernels/vector_add_b16.py`
- `kernels/saxpy_select_b16.py`
- `kernels/i32_add_select_b16.py`

These are direct TTIR analogues of the Phase 5 handwritten value elementwise hardware feature band: launch identity, lane ranges, masked loads, elementwise ALU, compare/select, and masked stores.

The future headline corpus keeps the roadmap honest:

- `kernels/future/softmax_row.py`
- `kernels/future/matmul_dot.py`
- `kernels/future/layer_norm_forward.py`

Those future kernels are included for inventory and planning. Their features are staged for later phases such as reductions, approximate math/SFU policy, dot/contract planning, and richer memory legality. They are not Phase 7 lowerable commitments.

TTIR snapshots under `generated/` are emitted from real Triton source by:

```bash
tools/vc4_emit_ttir.py ...
```

They are source-controlled for reproducibility and review. Regeneration is optional and requires the pinned Phase 6 Triton environment described in `compiler/docs/vc4_triton_frontend_setup.md`.

These snapshots are TTIR. They are not TTGIR, NVIDIA IR, `nvgpu`, `nvvm`, LLVM IR, PTX, cubin, or hsaco.

No normal VC4 build or `check-vc4` depends on Triton, PyTorch, CUDA, HIP, a GPU, or regenerating this corpus.
