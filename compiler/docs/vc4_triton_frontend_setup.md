# VC4 Triton Frontend Setup

Phase 6 establishes a pinned, optional Triton frontend path for generating and parsing real TTIR. It is limited to frontend setup, TTIR inventory, corpus generation, and importer skeleton work.

Phase 6 does not implement TTIR-to-value semantic lowering, does not lower TTIR to `vc4kernel`, does not ingest TTGIR or vendor GPU IR, and does not run hardware. The required eventual path remains:

```text
TTIR -> standard value layer -> vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware
```

READY_FOR_TRITON remains NO.

## Pinned Version

The Phase 6 source of truth is Triton 3.7.0:

```text
PyPI package: triton==3.7.0
Source tag:   v3.7.0
Source URL:   https://github.com/triton-lang/triton/tree/v3.7.0
```

Use the PyPI package when the local platform has a matching wheel:

```bash
python3 -m venv .vc4_auto/triton_phase6_venv
. .vc4_auto/triton_phase6_venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'triton==3.7.0'
```

If PyPI has no matching wheel for the local platform, use the pinned source fallback:

```bash
python3 -m venv .vc4_auto/triton_phase6_venv
. .vc4_auto/triton_phase6_venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
rm -rf .vc4_auto/triton_v3_7_0_src
git clone --depth 1 --branch v3.7.0 https://github.com/triton-lang/triton.git .vc4_auto/triton_v3_7_0_src
python -m pip install -r .vc4_auto/triton_v3_7_0_src/python/requirements.txt
python -m pip install -e .vc4_auto/triton_v3_7_0_src
```

This setup is optional. Normal `check-vc4` must not require Triton, PyTorch, CUDA, HIP, a GPU, generated TTIR snapshots, or network access.

## Why TTIR First

TTIR is the first producer IR because it is Triton's source-level MLIR representation before target-specific GPU lowering. Phase 6 intentionally does not ingest TTGIR, `ttnvgpu`, `nvgpu`, `nvvm`, `rocdl`, LLVM IR, PTX, cubin, or hsaco as the first producer path. Those forms are already target-specific or too low-level for the VC4 value-layer boundary.

Phase 6 tooling emits and parses TTIR only. Any future importer must first parse real TTIR through Triton's MLIR and dialect machinery. Text scans are allowed only after that parse succeeds, and only for inventory/reporting.

## Frontend Target

The frontend helpers use `cuda:80:32` as an explicit Triton target for AST-to-TTIR generation. This target supplies Triton backend codegen hooks, option parsing, module maps, and dialect registration needed by Triton's frontend API. Phase 6 stores only TTIR and does not emit TTGIR, PTX, cubin, or any GPU binary.

The target is therefore a frontend API requirement, not a claim that VC4 uses CUDA or that CUDA execution is involved.

## Optional Tools

The optional tools are:

```bash
tools/vc4_emit_ttir.py SOURCE.py \
  --kernel-name KERNEL \
  --signature '*fp32,*fp32,*fp32,i32,16' \
  --target cuda:80:32 \
  --num-warps 1 \
  --num-stages 3 \
  --out OUT.ttir.mlir \
  --metadata-out OUT.json

tools/vc4_parse_ttir.py INPUT.ttir.mlir \
  --target cuda:80:32 \
  --summary-json OUT.json
```

The emitter uses Triton 3.7.0 frontend-only `ASTSource.make_ir(...)`, not full `triton.compile(...)`. The parser uses Triton's MLIR parser and loaded Triton/backend dialects.

## References

- Main docs: https://triton-lang.org/main/index.html
- `program_id`: https://triton-lang.org/main/python-api/generated/triton.language.program_id.html
- `num_programs`: https://triton-lang.org/main/python-api/generated/triton.language.num_programs.html
- `arange`: https://triton-lang.org/main/python-api/generated/triton.language.arange.html
- `load`: https://triton-lang.org/main/python-api/generated/triton.language.load.html
- `store`: https://triton-lang.org/main/python-api/generated/triton.language.store.html
- `reduce`: https://triton-lang.org/main/python-api/generated/triton.language.reduce.html
- `dot`: https://triton-lang.org/main/python-api/generated/triton.language.dot.html
- Vector-add tutorial: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
- Fused-softmax tutorial: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
- Matmul tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- Layer-norm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
- PyTorch compilation-stages blog: https://pytorch.org/blog/triton-kernel-compilation-stages/

