# VC4 Real TTIR Inventory

Phase 6d locks the current inventory of real Triton 3.7.0 TTIR forms observed in the source-controlled Phase 6 corpus. This document summarizes generated TTIR snapshots under `examples/triton/phase6/generated/`; it is not a semantic importer, lowering plan, or hardware support claim.

READY_FOR_TRITON remains NO.

## Corpus

Required Phase 7 elementwise candidates:

- `vector_add_b16`: `tt.func`, axis-0 `tt.get_program_id`, `tt.make_range`, masked `tt.load`, `arith.addf`, masked `tt.store`.
- `saxpy_select_b16`: required elementwise forms plus `arith.mulf`, `arith.cmpf`, and `arith.select` from `tl.where`.
- `i32_add_select_b16`: required elementwise forms plus i32 `arith.addi`, `arith.subi`, `arith.cmpi`, and `arith.select`.

Future headline inventory:

- `softmax_row`: row masked load with `other=-inf`, `tt.reduce`, `math.exp`, `arith.divf`, and store. Approximate exp policy is staged for later math/SFU phases.
- `matmul_dot`: two-axis program IDs, rank-2 block tensor arithmetic, `tt.broadcast`, `tt.expand_dims`, masked loads, `tt.dot`, and masked store.
- `layer_norm_forward`: row masked loads, `tt.reduce`, `math.sqrt`, floating division, select, and store.

## Phase 7 Required Forms

The Phase 7 real TTIR elementwise smoke should require:

- `tt.func` and `tt.return` for the TTIR function boundary.
- axis-0 `tt.get_program_id` for launch identity.
- `tt.make_range` as the lowered form of `tl.arange`, mapped later to value-layer lane/range construction.
- `tt.splat` for scalar-to-block expansion.
- masked tensor-of-pointers `tt.load` with zero `other` values for the required elementwise kernels.
- masked tensor-of-pointers `tt.store`.
- `arith.addi`/`arith.cmpi` for offset and mask math.
- `arith.addf`, `arith.cmpf`, and `arith.select` where present in the f32 select candidates.

These forms target the standard value layer: `func`, tiny `vc4value`, `vector`, `memref`, `arith`, and later `math` as explicitly phased. They must not be lowered directly to `vc4kernel`.

## Staged And Future Forms

The matrix at `compiler/docs/vc4_ttir_frontend_inventory_matrix.json` records staged and future-profile forms observed or reserved by the corpus:

- `tt.reduce` and `tt.reduce.return` for future reductions.
- `math.exp` for future approximate math policy.
- `math.sqrt` as a later math policy input.
- `tt.dot` for future `vector.contract` planning.
- rank-2 block tensor forms such as `tt.broadcast` and `tt.expand_dims` in the matmul/dot snapshot.
- block pointer forms, atomics, cache modifiers, and volatile memory are not Phase 7-lowerable and remain reject/future-profile entries.

## Checker Discipline

The Phase 6d checker first parses each generated TTIR file through Triton's MLIR parser via `tools/vc4_parse_ttir.py`. Only after parse succeeds does it inventory op names from the parsed module summary. Text scanning is reporting only; it is not importer logic and not lowering.

Normal `check-vc4` does not depend on Triton, PyTorch, CUDA, HIP, a GPU, or regenerating TTIR.
