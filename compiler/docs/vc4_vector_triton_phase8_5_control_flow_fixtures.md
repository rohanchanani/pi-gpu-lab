PHASE85_CONTROLLED_TRITON_CF_FIXTURES=YES
REAL_TRITON_CF_SOURCES=YES
REAL_TTIR_CF_SNAPSHOTS=YES
ACCEPTED_FIXTURES_EXCLUDE_ARITH_SITOFP=YES
GENERIC_PHASE85A1_PROBES_ARE_INVENTORY_ONLY=YES
READY_FOR_PHASE85R3_IMPORTER_REGION_FRAMEWORK=YES
PHASE85_TTIR_REGION_LOWERING_FRAMEWORK=YES
PHASE85_CONTROLLED_TTIR_CF_STATIC_PIPELINE=PASS
VECTOR_INDEX_CAST_IMPORTER_BOILERPLATE_REMOVED=YES
ARITH_SITOFP_STAGED_BY_BODY_FEATURE=YES
FRONTEND_ROBUSTNESS_AUDIT=PASS
READY_FOR_PHASE85R4_STATIC_AUDITS_CONTRACT_LOCK=YES
READY_FOR_TRITON=NO

# VC4 Vector/Triton Phase 8.5 Controlled Control-Flow Fixtures

Phase 8.5R2 adds controlled real Triton source fixtures and real emitted TTIR
snapshots for scalar/coherent TTIR control flow. These fixtures replace the
generic Phase85a1 probes as the acceptance-driving inputs. The Phase85a1 probes
remain inventory evidence for real Triton emission shapes.

Tracked sources:

```text
examples/triton/phase8_5_control_flow/sources/
```

Tracked generated TTIR snapshots:

```text
examples/triton/phase8_5_control_flow/generated/
```

Tracked manifest:

```text
examples/triton/phase8_5_control_flow/manifest.json
```

## Fixture Set

- `ttir_cf_scalar_if_b16`: scalar runtime `if` / `else`, emitted as `scf.if`.
- `ttir_cf_tl_range_loop_b16`: runtime `tl.range`, emitted as `scf.for`.
- `ttir_cf_while_loop_b16`: scalar coherent `while`, emitted as `scf.while`.
- `ttir_cf_persistent_loop_b16`: pid-strided persistent loop skeleton using
  `tl.program_id(0)` and `tl.num_programs(0)`, emitted as `scf.while` with
  `tt.get_num_programs`.
- `mixed_ttir_cf_loop_if_tail_b16`: mixed tail-masked load/store, scalar
  `while`, scalar runtime `if`, f32 arithmetic, compare, and select.
- `ttir_cf_static_range_unrolled_b16`: optional static-range import regression;
  emitted as specialized/unrolled straight-line TTIR with no runtime `scf`.

## Accepted Body Features

The accepted control-flow fixtures intentionally avoid unrelated staged body
features. Bodies use only f32/i32 add, sub, mul, cmp/select, constants, splats,
`tl.arange(0, BLOCK_SIZE)`, contiguous masked `tl.load`/`tl.store`, and tail
masks.

The emitted accepted snapshots contain:

```text
ACCEPTED_FIXTURES_CONTAIN_ARITH_SITOFP=NO
ACCEPTED_FIXTURES_CONTAIN_UNRELATED_STAGED_BODY_FEATURES=NO
```

Rejected from this fixture band:

```text
arith.sitofp
fptosi
dot/reduce
gather/scatter
block pointers / tt.make_block_ptr / tt.advance
atomics
subword features
SFU/math expansion
TTGIR / TritonGPU / NVGPU / NVVM / GPU backend dialects
```

## Inventory Result

The controlled inventory is recorded in:

```text
.vc4_auto/vector_triton_phase85r2_controlled_snapshots/inventory/CONTROLLED_TTIR_CF_INVENTORY.md
.vc4_auto/vector_triton_phase85r2_controlled_snapshots/json/controlled_ttir_cf_inventory.json
```

Locked classifications:

```text
SCALAR_IF_FIXTURE_FORM=SCF_IF
TL_RANGE_FIXTURE_FORM=SCF_FOR
WHILE_FIXTURE_FORM=SCF_WHILE
PERSISTENT_FIXTURE_FORM=SCF_WHILE
MIXED_FIXTURE_FORM=SCF_CF
STATIC_RANGE_FIXTURE_FORM=UNROLLED_OR_SPECIALIZED
```

## Importer Region Lowering Lock

Phase 8.5R3 implements structural TTIR-to-VC4Value region lowering for the
controlled snapshots above. `tt.func` lowers to `func.func`, and nested
`scf.for`, `scf.if`, and `scf.while` regions are copied into the value surface
with scoped SSA mappings for block arguments and loop-carried values.

The controlled snapshots statically reach scheduled VC4 through:

```text
vc4-triton-opt --convert-triton-to-vc4-value
vc4-opt --vc4-verify-value-surface
vc4-opt --convert-scf-to-cf --convert-vc4-value-to-vc4kernel --verify-vc4kernel
vc4-opt --convert-vc4kernel-to-ssavc4
vc4-opt --convert-ssavc4-to-vc4
```

`tt.make_range` is consumed structurally for canonical pointer and tail-mask
expressions, so the importer no longer emits dead common-prefix
`vector<16xindex>` to `vector<16xi32>` cast boilerplate. If `tt.make_range`
is used as a true data value, it is handled separately from the canonical
transfer path.

`arith.sitofp` remains staged as an unrelated numeric-cast body feature. It is
not part of the accepted control-flow fixture body set.

The importer continues to emit value-layer IR only and does not emit or preserve
TTIR, backend, `vc4kernel`, `ssavc4`, or scheduled `vc4` operations.

`READY_FOR_TRITON` remains `NO`.
