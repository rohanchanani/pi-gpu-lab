# VC4Tile M5 Copy Planner Design: Tile Movement, VPM/VDR/VDW/TMU Selection, and Ergonomic Data-Movement Surface

**Date:** 2026-05-27  
**Status:** locked design decision for the M5 copy-planner big-ticket item  
**Scope:** ergonomic VC4Tile data movement, tile views/layouts, copy planning, VPM shared-memory movement, global-memory movement, and lowering to VC4Tile core / SSAVC4 / scheduled VC4  
**Non-scope:** Triton lowering, IREE lowering, real sub-32 precision lowering, performance tuning beyond correctness-first path selection, and final low-level QASM scheduling details  
**Companion design documents:**

- `vc4tile_m5_compute_primitives_design.md` / `vc4tile_m5_compute_primitives_design(1).md`
- `vc4tile_precision_roadmap_design.md` / `vc4tile_precision_roadmap_design(1).md`
- `vc4tile_m5_full_design.md`

---

## 1. Original question

We needed to decide how the M5 ergonomic VC4Tile layer should expose data movement and how that data movement should map onto the actual Raspberry Pi VideoCore IV hardware.

The concrete question was:

> What is the natural copy-planner design for VC4Tile? It needs to expose the tile movement primitives that make sense for M5 and later producer lowering, while also efficiently lowering to how the Pi GPU actually moves data: TMU, VPM, VDW, VDR/VCD, QPU reads/writes, DMA setup, and shared-memory/barrier constraints.

This question is not a generic “should we add load/store?” question. VC4 has a very particular memory hierarchy:

```text
global memory
  -> TMU direct per-lane read path
  -> VCD/VDR DMA load into VPM path
VPM / shared memory
  -> QPU VPM horizontal/vertical read/write path
  -> VDW DMA store out to global path
register / lane vectors
  -> QPU ALU and VPM write path
```

The copy planner is the bridge between a CuTe/ThunderKittens-like tile-programming surface and the concrete VC4 mechanisms. It must let kernel authors and future producer-lowering passes say “load this tile,” “copy this tile to shared,” “store this transposed tile,” or “materialize this view,” without making them spell VPM setup words or VDW/VDR register encodings.

---

## 2. High-level decision

M5 should add a **copy-planning pass** as the first substantive ergonomic implementation block after the surface/core pipeline and tile/layout type vocabulary are in place.

The pass should be named, unless package generation chooses a mechanically more consistent spelling:

```text
--plan-vc4tile-copies
```

It should run in this pipeline order:

```text
vc4tile ergonomic surface input
  -> --canonicalize-vc4tile-surface
  -> --plan-vc4tile-copies
  -> --legalize-vc4tile-core-cfg
  -> --verify-vc4tile-core
  -> --convert-vc4tile-to-ssavc4
  -> --convert-ssavc4-to-vc4
  -> vc4-codegen --emit-bundle
  -> hardware
```

The copy planner should consume ergonomic tile movement operations:

```text
vc4tile.tile_load
vc4tile.tile_store
vc4tile.copy_tile
vc4tile.tile_view
vc4tile.tile_subview
vc4tile.transpose_view
vc4tile.shared_tile_alloc
```

and produce only legal VC4Tile core operations and lower-half-compatible metadata. By the time `--verify-vc4tile-core` runs, no surface-only tile movement op may remain.

The planner must be **hardware-aware but not hardware-exposing**:

```text
Surface:   tile_load / tile_store / copy_tile / transpose_view / shared_tile_alloc
Planner:   choose TMU, VPM read/write, VDW, VDR/VCD, barriers, masks, row allocation
Core:      masked_load_global, masked_store_global, shared_load/store, arithmetic, cf/block args
SSAVC4:    tmu, vpm.read/write, vdw.store, future vdr.load, barrier/sema, branch
VC4:       scheduled qpu bundles/pseudoops/register-mapped I/O
```

---

## 3. Why the copy planner must be first in M5

The compute-primitives decision makes `tile_contract`, `tile_dot`, reductions, and companion movement ops producer-facing targets. Those compute primitives are only useful if their inputs and outputs can be described as tiles with layouts, memory spaces, and copy strategies. The precision roadmap also concludes that future sub-32 work is primarily a storage/copy-planning problem first, not a tensor-core-like compute problem.

Therefore M5 should not add dot/matmul first. It should first establish:

```text
tile identity
layout vocabulary
memory-space vocabulary
shared VPM allocation model
copy planning
hardware-proven global/shared/register movement
```

After that, elementwise compute, reductions, and contractions can target concrete tile fragments and rely on the same planner.

Without a copy planner, every future feature would risk re-implementing ad hoc movement logic:

```text
shared transpose     -> own custom VPM logic
tile_dot             -> own custom load/store logic
Triton lowering      -> own custom coalesced load/store logic
IREE lowering        -> own custom promotion/copy logic
sub-32 precision     -> own custom pack/unpack/load/store logic
```

That is exactly what M5 should avoid.

---

## 4. Hardware evidence and constraints

### 4.1 QPU lane model

VC4 QPUs are a 16-way virtual SIMD machine. For M5, the fundamental register tile fragment is therefore a 16-lane vector:

```text
vector<16xi32>
vector<16xf32>
vector<16xi1> mask
```

The copy planner should view one QPU request/warp as owning a 16-lane vector row unless an operation explicitly works through cooperative shared VPM state across multiple warps.

### 4.2 VPM shared-memory model

From the QPU perspective, VPM is a 2D array of 32-bit words:

```text
width:      16 words
max height: 64 rows
word size:  32 bits
```

For M5 executable paths, this means the planner should treat the visible user VPM window as a 64 × 16 × 4-byte shared-memory region. M5 remains 32-bit-only, so every planned shared tile cell is one 32-bit word.

Future sub-32 support can use the VPM 8-bit/16-bit packed/laned modes, but M5 must reject those modes rather than silently treating them as 32-bit.

### 4.3 VPM QPU read/write modes

The QPU can read and write VPM horizontal or vertical 16-way vectors. This is the natural mechanism for:

```text
register -> shared
shared -> register
shared transpose by horizontal write + vertical read, or vertical write + horizontal read
```

In 32-bit M5, the planner only needs the 32-bit horizontal/vertical modes.

### 4.4 VDW store path

The VDW path stores VPM data to global memory using DMA-style setup. This is the natural mechanism for:

```text
register -> global
shared -> global
```

because VC4 does not have a normal register-to-global store instruction. The correct store path is:

```text
register tile
  -> VPM write
  -> VDW store
  -> global memory
```

Any ergonomic `tile_store` that stores register data to global memory must eventually lower through VPM and VDW, not pretend that register-to-global exists.

### 4.5 VDR/VCD load path

The VDR/VCD path loads 2D global-memory blocks into VPM. This is the natural mechanism for:

```text
global -> shared
regular 2D global tile -> VPM shared tile
future packed/sub-32 loads into VPM
```

Current SSAVC4 support has VDW store, VPM read/write, TMU load, and barrier support. It does not yet have the symmetric VDR/VCD DMA load abstraction at the same level as VDW. Therefore M5 should add VDR/VCD support early, before the copy planner depends on it for global-to-shared movement.

### 4.6 TMU direct load path

The TMU path is the existing direct global-memory load mechanism. It is useful for:

```text
global -> register vector loads
coalesced independent-vector loads
some strided/scattered per-lane loads
fallback when VDR cannot express the requested global->shared copy
```

The planner should prefer VDR for regular 2D global-to-shared movement once VDR exists, but it should keep TMU as the direct register-load path and as the correctness fallback for patterns VDR cannot express.

---

## 5. Required surface concepts

The copy planner depends on a tile vocabulary. M5 should add this vocabulary before or alongside the planner.

### 5.1 Tile descriptor fields

Every tile movement operation must carry, directly or indirectly, these fields:

```text
logical_shape        static rank/extent, e.g. [16], [rows, 16], [m, n]
element_type         f32/i32/u32 in M5 executable paths
storage_type         f32/i32/u32 in M5 executable paths
memory_space         global | shared_vpm | register | immediate/splat
layout               row_major | col_major | affine_2d | vpm_row | vpm_col | transposed_view
strides              static affine strides where applicable
offset_unit          byte | element
alignment            known minimum alignment in bytes, if known
boundary_policy      exact | tail_predicated | zero | clamp | reject for unsupported
mask                 vector<16xi1> or derived tail mask when vectorized
role                 input | output | accumulator | scratch | metadata-only if not yet consumed
precision_policy     exact_32 in M5; future sub-32 policies rejected
packing              none in M5; future pack modes rejected
```

The exact MLIR spelling can be finalized during package generation, but the semantic fields are required. A suggested canonical attr family is:

```mlir
#vc4tile.layout<row_major>
#vc4tile.layout<col_major>
#vc4tile.layout<affine_2d, strides = [64, 4], offset_unit = byte>
#vc4tile.layout<vpm_row>
#vc4tile.layout<vpm_col>
#vc4tile.layout<transposed_view>

#vc4tile.memory_space<global>
#vc4tile.memory_space<shared_vpm>
#vc4tile.memory_space<register>

#vc4tile.boundary<exact>
#vc4tile.boundary<tail_predicated>
#vc4tile.boundary<zero>
#vc4tile.boundary<clamp>

#vc4tile.precision<storage = f32, expressed = f32, accumulator = f32, packing = none>
```

For M5, unsupported precision/packing fields must fail deterministically.

### 5.2 `vc4tile.tile_load`

Purpose:

```text
Load a logical tile from global or shared memory into a register tile or shared tile.
```

Recommended forms:

```mlir
%tile = vc4tile.tile_load %base[%row, %col], %mask
  { shape = [1, 16],
    memory_space = #vc4tile.memory_space<global>,
    layout = #vc4tile.layout<row_major>,
    elem_type = f32,
    storage_type = f32,
    precision = #vc4tile.precision<exact_32>,
    boundary = #vc4tile.boundary<tail_predicated> }
  : (i32, i32, i32, vector<16xi1>) -> !vc4tile.tile<1x16xf32, register>
```

Allowed M5 executable sources:

```text
global -> register
global -> shared_vpm, after VDR/VCD support exists
shared_vpm -> register
```

Rejected in M5:

```text
sub-32 storage types
rank/dynamic layout not handled by the planner
boundary policies not implemented
runtime-dynamic shared-memory allocation
unbounded arbitrary affine maps
```

### 5.3 `vc4tile.tile_store`

Purpose:

```text
Store a register or shared tile to global or shared memory.
```

Recommended form:

```mlir
vc4tile.tile_store %tile, %base[%row, %col], %mask
  { shape = [1, 16],
    memory_space = #vc4tile.memory_space<global>,
    layout = #vc4tile.layout<row_major>,
    elem_type = i32,
    storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    boundary = #vc4tile.boundary<tail_predicated> }
  : !vc4tile.tile<1x16xi32, register>, i32, i32, i32, vector<16xi1>
```

Allowed M5 executable destinations:

```text
register -> global, through VPM write + VDW store
shared_vpm -> global, through VDW store
register -> shared_vpm, through VPM write
```

Important invariant:

```text
The planner must never lower register -> global as a direct store.
It must produce or cause the lower half to produce register -> VPM -> VDW -> global.
```

### 5.4 `vc4tile.copy_tile`

Purpose:

```text
Copy between tile locations while preserving or transforming layout.
```

Recommended form:

```mlir
%dst = vc4tile.copy_tile %src
  { src_space = #vc4tile.memory_space<global>,
    dst_space = #vc4tile.memory_space<shared_vpm>,
    src_layout = #vc4tile.layout<row_major>,
    dst_layout = #vc4tile.layout<vpm_row>,
    shape = [16, 16],
    elem_type = f32,
    precision = #vc4tile.precision<exact_32> }
  : !vc4tile.tile<16x16xf32, global> -> !vc4tile.tile<16x16xf32, shared_vpm>
```

This operation is the planner’s central abstraction. Depending on source/destination/layout, it can lower to different atoms:

```text
global -> register:        TMU load
global -> shared_vpm:      VDR/VCD load if regular, else TMU + VPM write fallback
register -> shared_vpm:    VPM write
shared_vpm -> register:    VPM read
shared_vpm -> global:      VDW store
register -> global:        VPM write + VDW store
register -> register:      SSA value / reshape / vector.select where legal
shared transpose:          VPM orientation change
```

### 5.5 `vc4tile.tile_view` and `vc4tile.tile_subview`

Purpose:

```text
Create a symbolic view of a tile without moving data.
```

Views are surface-only. The planner/canonicalizer must eliminate them by folding offsets/layout composition into `tile_load`, `tile_store`, or `copy_tile`.

Recommended form:

```mlir
%sub = vc4tile.tile_subview %tile
  { offsets = [0, 0], sizes = [16, 16], strides = [1, 1] }
  : !vc4tile.tile<32x32xf32, global> -> !vc4tile.tile<16x16xf32, global>
```

Verifier requirements:

```text
sizes and strides must be static in M5
result rank must be <= source rank
no dynamic affine maps in M5
no view may remain after --canonicalize-vc4tile-surface / --plan-vc4tile-copies
```

### 5.6 `vc4tile.transpose_view`

Purpose:

```text
Create a symbolic transposed view without immediately moving data.
```

Recommended form:

```mlir
%t = vc4tile.transpose_view %tile { permutation = [1, 0] }
  : !vc4tile.tile<16x16xf32, shared_vpm> -> !vc4tile.tile<16x16xf32, shared_vpm>
```

Planner lowering:

```text
shared_vpm transpose view consumed by register load:
  choose opposite-orientation VPM read

register/global transpose materialization:
  choose VPM write/read orientation transform when shape fits

shared_vpm transpose view consumed by global store:
  choose VPM/VDW orientation if expressible; otherwise materialize register path or reject
```

### 5.7 `vc4tile.shared_tile_alloc`

Purpose:

```text
Declare static VPM-backed shared tile storage.
```

Recommended form:

```mlir
%smem = vc4tile.shared_tile_alloc
  { shape = [16, 16], elem_type = f32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch> }
  : !vc4tile.tile<16x16xf32, shared_vpm>
```

M5 restrictions:

```text
static shape only
element size = 4 bytes only
allocation must fit vpm_rows_per_block / vpm_bytes_per_block
no dynamic shared memory
no sub-32 packing
no more than one simple static VPM allocation per early slice unless resource accounting is implemented
```

---

## 6. Planner output model

The planner should not output scheduled VC4 or SSAVC4 directly. It should output legal VC4Tile core operations. Those core ops then lower through the existing and extended lower half.

### 6.1 Existing core operations to reuse

M4 already has core-ish operations in this family:

```text
vc4tile.program_id / block_id / warp_id / lane_id / lane_range
vc4tile.mask_all
vc4tile.tail_mask
vc4tile.masked_load_global
vc4tile.masked_store_global
vc4tile.shared_alloc
vc4tile.shared_load
vc4tile.shared_store
vc4tile.rotate
vc4tile.reduce
vc4tile.barrier
arith.*
cf.br / cf.cond_br after SCF legalization
vc4tile.return
```

The planner should rewrite ergonomic movement ops into these where possible.

### 6.2 Possible new core copy atoms

If existing M4 core ops are not precise enough, M5 may add small core-only atoms. These atoms must still be lower-level than ergonomic surface ops and must be legal after `--verify-vc4tile-core`.

Recommended candidates:

```text
vc4tile.vpm_read_tile
vc4tile.vpm_write_tile
vc4tile.vdr_load_tile
vc4tile.vdw_store_tile
```

However, adding these is optional and should only happen if they make core verification/lowering clearer. The preferred first approach is to lower to existing shared/global core ops and extend their attributes if sufficient.

### 6.3 Required core invariants

After copy planning and core CFG legalization:

```text
No vc4tile.tile_load
No vc4tile.tile_store
No vc4tile.copy_tile
No vc4tile.tile_view
No vc4tile.tile_subview
No vc4tile.transpose_view
No scf.*
No index-typed values
No producer dialect ops
Only VC4Tile core ops, arith, cf, and accepted builtin ops remain
All shared memory resource usage is statically recorded on the kernel
All VPM rows/bytes are statically accounted
All sub-32 metadata is default exact_32/none or rejected
```

---

## 7. Planning algorithm

### 7.1 Inputs

For each copy-like surface op, the planner gathers:

```text
source memory space
destination memory space
source layout
destination layout
logical shape
element/storage type
mask / boundary policy
static offsets and strides
known alignment
kernel schedule mode
shared VPM allocation/resource state
barrier/resource requirements
```

### 7.2 Candidate strategy enumeration

For each op, the planner enumerates candidate strategies. Example:

```text
copy_tile(global,row_major -> shared_vpm,vpm_row, shape=[16,16], f32):
  candidate A: VDR_LOAD_2D_32
  candidate B: for each row: TMU_VECTOR_LOAD_32 + VPM_WRITE_ROW_32
  candidate C: reject if neither works
```

For global-to-register:

```text
copy_tile(global,row_major -> register, shape=[1,16], f32):
  candidate A: TMU_VECTOR_LOAD_32
  candidate B: VDR_LOAD_2D_32 + VPM_READ_ROW_32, only if reuse/shared staging requested
```

For register-to-global:

```text
copy_tile(register -> global,row_major, shape=[1,16], i32):
  candidate A: VPM_WRITE_ROW_32 + VDW_STORE_2D_32
```

For shared transpose:

```text
copy_tile(shared_vpm,row_major -> register,transposed_view, shape=[16,16], f32):
  candidate A: VPM_READ_COL_32 after row-major shared write
```

### 7.3 Strategy scoring

M5 should use deterministic correctness-first scoring, not clever performance heuristics.

Recommended score order:

```text
1. Exact hardware-supported 2D VDR/VDW strategy with no scalarization
2. Direct TMU register load for simple register destination
3. VPM orientation transform for transpose/shared copies
4. TMU + VPM fallback for global->shared if VDR cannot express the layout
5. Reject with diagnostic
```

This means M5 can be simple but deterministic. Later milestones can improve scoring.

### 7.4 Validation and diagnostics

The planner must not silently approximate unsupported patterns. If the planner cannot exactly implement the requested copy, it must emit a diagnostic explaining which field failed:

```text
error: vc4tile.copy_tile cannot plan global->shared copy: non-static row stride is unsupported in M5
error: vc4tile.tile_load requires 32-bit storage in M5; got storage_type = f16
error: vc4tile.transpose_view of rank 3 is unsupported in M5
error: vc4tile.tile_store register->global requires VDW-capable 32-bit layout; got unsupported affine layout
```

The diagnostic should include:

```text
operation name
source/destination memory spaces
layout name
shape
specific unsupported property
```

---

## 8. Concrete strategy table

| Source | Destination | Layout pattern | M5 primary strategy | M5 fallback | Notes |
|---|---|---|---|---|---|
| global | register | 1×16 contiguous/coalesced | TMU vector load | none | Existing global load path. |
| global | register | 1×16 affine stride | TMU strided/per-lane load if existing core supports it | reject | Avoid pretending VDR is register-load. |
| register | global | 1×16 contiguous/tail | VPM write row + VDW store | none | No direct register->global store. |
| register | shared | 1×16 row | VPM write horizontal | none | Static row allocation. |
| shared | register | 1×16 row | VPM read horizontal | none | Static row allocation. |
| shared | register | 1×16 col / transpose | VPM read vertical | none | Core for ergonomic transpose. |
| global | shared | regular 2D row-major | VDR/VCD DMA load | TMU rows + VPM writes | Requires VDR support. |
| shared | global | regular 2D row-major | VDW DMA store | VPM read + VPM write/store if needed | Existing VDW store path. |
| register | register | same layout | SSA alias | vector shuffle/select if available | No hardware movement. |
| register | shared transpose | 16×16 materialization | VPM horizontal writes, vertical reads later | reject if shape not fitting VPM | Used by shared transpose. |
| global | global | arbitrary copy | not a primitive in M5 | reject | Use load/copy/store through tiles. |

---

## 9. VDR/VCD addition required by this design

### 9.1 Why VDR is required

Without VDR/VCD load support, global-to-shared tile copies must be expressed as repeated TMU register loads followed by VPM writes. That is correct for some shapes, but it misses the natural VC4 hardware path for regular 2D loads.

Because VDW store already exists in the lower half, adding the load-side counterpart is the cleanest way to make the copy planner hardware-complete for 32-bit tile movement.

### 9.2 Recommended SSAVC4 operation

Recommended spelling:

```mlir
ssavc4.vdr.load %base
  { elem_bytes = 4 : i32,
    row_len = 16 : i32,
    nrows = 16 : i32,
    memory_pitch_bytes = 64 : i32,
    vpm_base_row = 0 : i32,
    vpm_base_col = 0 : i32,
    orientation = #ssavc4.vpm_orientation<horizontal>,
    vpitch = 16 : i32,
    serialize = "mutex" }
  : i32
```

Exact field names may follow existing VDW naming conventions, but the semantics must include:

```text
global base address
row length
number of rows
memory pitch
VPM base row/column
orientation
width = 32-bit in M5
serialization / wait behavior
```

### 9.3 Required tests for VDR

Dialect / conversion tests:

```text
compiler/test/Dialect/SSAVC4/vdr-load-roundtrip.mlir
compiler/test/Dialect/SSAVC4/vdr-load-invalid.mlir
compiler/test/Conversion/SSAVC4ToVC4/vdr-load.mlir
compiler/test/CodeGen/SSAVC4/Emit/vdr-load-ssavc4.mlir
```

Hardware fixture:

```text
compiler/test/CodeGen/SSAVC4/Hardware/Run/vdr_load_roundtrip_ssavc4/input.mlir
compiler/test/CodeGen/SSAVC4/Hardware/Run/vdr_load_roundtrip_ssavc4/expected.json
compiler/test/CodeGen/SSAVC4/Hardware/Run/vdr_load_roundtrip_ssavc4/candidate/vdr_load_roundtrip_ssavc4_candidate_harness.c
```

Expected behavior:

```text
Host initializes a 2D source tile in global memory.
QPU VDR-loads it into VPM.
QPU reads VPM rows/columns and writes a transformed or copied result back through VDW.
Harness copies back and validates every element plus sentinels.
```

---

## 10. Copy planner lowering examples

### 10.1 One-row global load/store

Surface:

```mlir
vc4tile.kernel @tile_load_store_1d(%out : i32, %in : i32, %n : i32)
  attributes {
    public_name = "tile_load_store_1d",
    schedule_mode = #vc4tile.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
      {name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "u32"}
    ]
  } {
^entry(%out : i32, %in : i32, %n : i32):
  %zero = arith.constant 0 : i32
  %mask = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %tile = vc4tile.tile_load %in[%zero], %mask
    {shape = [1, 16], layout = #vc4tile.layout<row_major>,
     memory_space = #vc4tile.memory_space<global>, elem_type = i32,
     storage_type = i32, precision = #vc4tile.precision<exact_32>}
    : (i32, i32, vector<16xi1>) -> !vc4tile.tile<1x16xi32, register>
  vc4tile.tile_store %tile, %out[%zero], %mask
    {shape = [1, 16], layout = #vc4tile.layout<row_major>,
     memory_space = #vc4tile.memory_space<global>, elem_type = i32,
     storage_type = i32, precision = #vc4tile.precision<exact_32>}
    : !vc4tile.tile<1x16xi32, register>, i32, i32, vector<16xi1>
  vc4tile.return
}
```

Planned core shape:

```mlir
%lanes = vc4tile.lane_range : vector<16xi32>
%mask = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
%ld = vc4tile.masked_load_global %in, %lanes, %mask
  {elem_bytes = 4 : i32, offset_unit = #vc4tile.offset_unit<byte>,
   memory_space = #vc4tile.memory_space<global>, access = #vc4tile.memory_access<coalesced>}
  : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
vc4tile.masked_store_global %out, %lanes, %ld, %mask
  {elem_bytes = 4 : i32, offset_unit = #vc4tile.offset_unit<byte>,
   memory_space = #vc4tile.memory_space<global>, access = #vc4tile.memory_access<affine_contiguous>}
  : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
```

### 10.2 Shared transpose ergonomic surface

Surface:

```mlir
vc4tile.kernel @shared_transpose_ergonomic(%out : i32, %in : i32)
  attributes {
    public_name = "shared_transpose_ergonomic",
    schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
    uses_shared_vpm = true,
    uses_barrier = true,
    warps_per_block_max = 12 : i32,
    vpm_rows_per_block = 16 : i32,
    vpm_bytes_per_block = 1024 : i32
  } {
^entry(%out : i32, %in : i32):
  %tile = vc4tile.tile_load %in[0, 0]
    {shape = [16, 16], memory_space = #vc4tile.memory_space<global>,
     layout = #vc4tile.layout<row_major>, elem_type = i32,
     precision = #vc4tile.precision<exact_32>}
    : (i32) -> !vc4tile.tile<16x16xi32, global>
  %smem = vc4tile.shared_tile_alloc
    {shape = [16, 16], elem_type = i32, layout = #vc4tile.layout<vpm_row>}
    : !vc4tile.tile<16x16xi32, shared_vpm>
  vc4tile.copy_tile %tile, %smem
    {src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>}
    : !vc4tile.tile<16x16xi32, global>, !vc4tile.tile<16x16xi32, shared_vpm>
  vc4tile.barrier
  %t = vc4tile.transpose_view %smem {permutation = [1, 0]}
    : !vc4tile.tile<16x16xi32, shared_vpm> -> !vc4tile.tile<16x16xi32, shared_vpm>
  vc4tile.tile_store %t, %out[0, 0]
    {memory_space = #vc4tile.memory_space<global>, layout = #vc4tile.layout<row_major>,
     elem_type = i32, precision = #vc4tile.precision<exact_32>}
    : !vc4tile.tile<16x16xi32, shared_vpm>, i32
  vc4tile.return
}
```

Planned intent:

```text
1. VDR or TMU+VPM writes load global tile into VPM rows.
2. Barrier makes shared tile visible across participating warps.
3. Transpose view is represented by changed VPM read/store orientation, not by a massive mechanical surface program.
4. VDW stores transposed VPM data to global memory.
```

This is the canonical replacement for the current mechanical shared-transpose fixture.

---

## 11. Verification plan

M5 copy planner verification must be strong and hardware-backed. It should not rely on textual shape alone.

### 11.1 Dialect tests

Required dialect tests:

```text
compiler/test/Dialect/VC4Tile/tile-types-layouts-roundtrip.mlir
compiler/test/Dialect/VC4Tile/tile-types-layouts-invalid.mlir
compiler/test/Dialect/VC4Tile/tile-load-store-roundtrip.mlir
compiler/test/Dialect/VC4Tile/copy-tile-roundtrip.mlir
compiler/test/Dialect/VC4Tile/tile-view-roundtrip.mlir
compiler/test/Dialect/VC4Tile/tile-view-invalid.mlir
compiler/test/Dialect/VC4Tile/precision-markers-invalid.mlir
```

These tests must prove:

```text
attributes/types parse and print deterministically
unsupported non-32 precision is rejected
unsupported dynamic layouts are rejected
surface ops are not accepted as core after --verify-vc4tile-core
```

### 11.2 Lowered-IR tests

Required conversion tests:

```text
compiler/test/Conversion/VC4TileToSSAVC4/plan-copy-global-register.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-copy-register-global.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-copy-global-shared-vdr.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-copy-global-shared-tmu-fallback.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-copy-shared-register.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-copy-register-shared.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-transpose-view.mlir
compiler/test/Conversion/VC4TileToSSAVC4/reject-unplanned-copy-before-core.mlir
compiler/test/Conversion/VC4TileToSSAVC4/reject-sub32-copy-m5.mlir
```

These tests must prove:

```text
surface copy ops disappear after --plan-vc4tile-copies
planned core ops are legal under --verify-vc4tile-core
raw surface ops are rejected by --convert-vc4tile-to-ssavc4 if the planner was not run
expected SSAVC4 memory ops appear after full conversion
```

### 11.3 Scheduled/artifact tests

Required emit tests:

```text
compiler/test/CodeGen/VC4Tile/Emit/tile-load-store-vc4tile.mlir
compiler/test/CodeGen/VC4Tile/Emit/tile-copy-global-shared-vc4tile.mlir
compiler/test/CodeGen/VC4Tile/Emit/shared-transpose-ergonomic-vc4tile.mlir
compiler/test/CodeGen/VC4Tile/Emit/scf-tiled-copy-vc4tile.mlir
```

These tests must prove the full path:

```text
vc4tile surface
  -> planned vc4tile core
  -> ssavc4
  -> scheduled vc4
  -> manifest/bundle shape
```

### 11.4 Hardware fixtures

M5 should add at least these copy-planner hardware fixtures:

```text
tile_load_1d_tail_vc4tile
tile_store_1d_tail_vc4tile
tile_load_store_2d_row_major_vc4tile
tile_load_store_affine_stride_vc4tile
vdr_load_roundtrip_vc4tile
global_to_shared_to_global_2d_vc4tile
register_to_shared_roundtrip_vc4tile
shared_to_register_roundtrip_vc4tile
shared_transpose_16x16_ergonomic_vc4tile
shared_transpose_store_global_vc4tile
scf_tiled_copy_loop_vc4tile
```

Every fixture must:

```text
start from checked-in VC4Tile ergonomic input.mlir
run through the support runner pipeline with fresh generation by default
assemble/build/run candidate artifacts
copy back device output
compare against a CPU oracle in the harness
check sentinel regions
derive VC4_TEST_RESULT status from real mismatch/launch counters
avoid fixed PASS and synthetic runtime counters
```

### 11.5 Anti-shortcut scans

The copy planner slice and M5 final acceptance must scan for:

```text
no direct vc4tile -> vc4 pass
no producer lowering from Triton/IREE/JAX/PyTorch/StableHLO/gpu
no fixture-name special casing
no fake VC4_TEST_RESULT in support runners or candidate harnesses
no checked-in generated .vc4_auto outputs
no surface copy ops left in core lowering tests
no sub-32 executable lowering despite metadata fields
```

---

## 12. Relationship to compute primitives

The compute-primitives design locks the decision that `tile_contract`, `tile_dot`, `tile_matmul`, `tile_reduce`, and `block_reduce` belong in the ergonomic VC4Tile surface because they are natural targets for both future Triton and future IREE/Linalg lowering.

The copy planner is their companion and prerequisite.

The intended interplay is:

```text
tile_load      fetch operands into register/shared tiles
copy_tile      stage/reorder/promote operands
transpose_view express layout transforms without mechanical expansion
tile_contract  consume register/shared tile operands and produce accumulator/output tiles
tile_reduce    reduce tile axes using lane rotate/shared/barrier mechanisms
tile_store     write final tiles to global through VPM/VDW
```

`tile_contract` should not own global-memory access. It should consume tile values. The copy planner should own memory movement. This separation is what allows both Triton and IREE/Linalg lowering to reuse the same tile movement layer.

---

## 13. Relationship to precision roadmap

M5 is 32-bit-only in executable semantics. The copy planner must still preserve future precision fields because future sub-32 support is mostly a storage/layout/copy-planning problem.

M5 accepted precision:

```text
storage_type = f32/i32/u32
expressed_type = f32/i32/u32
accumulator_type = f32/i32/u32
precision_policy = exact_32
packing = none
```

M5 rejected precision:

```text
f16 storage
bf16 storage
fp8/fp4
int8/uint8/int4 storage
packed VPM sub-vectors
laned VPM sub-vectors
quantization scale/zero-point semantics
```

The copy planner should attach enough information to diagnostics and metadata so later precision milestones can add:

```text
VPM packed/laned 8/16-bit modes
VDW/VDR packed byte/halfword modes
regfile-A pack/unpack selection
explicit dequantize/quantize movement
storage f16 / compute f32 paths
specialized packed-byte ALU paths
```

But none of that is executable in M5.

---

## 14. Risks and mitigations

### Risk: the planner becomes a hidden direct lowering path

Mitigation:

```text
The planner emits VC4Tile core only.
It must not create scheduled VC4 ops.
It must not call SSAVC4 lowering directly.
Final acceptance scans for direct vc4tile -> vc4 shortcuts.
```

### Risk: tile movement ops become decorative and not consumed

Mitigation:

```text
--verify-vc4tile-core rejects every surface movement op.
--convert-vc4tile-to-ssavc4 rejects raw surface ops with an ordering diagnostic.
Every hardware fixture must start from surface tile movement ops, not preplanned core.
```

### Risk: VDR support expands M5 too much

Mitigation:

```text
Add VDR as a narrow vertical slice with one roundtrip hardware fixture.
If VDR is not complete by the copy-planner slice, global->shared can temporarily use TMU+VPM fallback, but M5 final should still include VDR or explicitly move it to a follow-up lower-half enabler before producer lowering.
```

### Risk: unsupported layout silently lowers incorrectly

Mitigation:

```text
Planner emits deterministic diagnostics.
Invalid tests cover dynamic stride, unsupported affine layout, unsupported rank, unsupported precision, and unaligned unsupported copy.
```

### Risk: hardware fixtures validate only constants

Mitigation:

```text
Harness oracles must depend on initialized input buffers, logical IDs, lanes, loop indices, and sentinel regions.
Expected output must not be a constant independent of the copied data.
```

---

## 15. Acceptance criteria for this design item

The copy planner design is accepted when M5 contains:

```text
1. Surface tile movement ops and layout/precision metadata.
2. --plan-vc4tile-copies pass.
3. --verify-vc4tile-core rejection of unplanned surface ops.
4. Planned core output for global/register/shared/tail/transpose cases.
5. VDR/VCD support or explicitly verified fallback with a tracked VDR follow-up gate.
6. At least ten substantive hardware fixtures covering the movement matrix.
7. Fresh candidate-first runner integration.
8. No direct vc4tile -> vc4 shortcut.
9. No producer lowering.
10. No executable sub-32 precision.
11. M2/M3/M4 regressions remain green.
```

---

## 16. Final locked decision

M5 will implement a VC4Tile copy planner as a dedicated pass that converts ergonomic tile movement operations into legal VC4Tile core and lower-half-compatible memory operations. The planner is the first major ergonomic implementation block because it underlies shared transpose, reductions, dot/contract, future Triton/IREE lowering, and future sub-32 precision support.

The planner will expose tile movement through `tile_load`, `tile_store`, `copy_tile`, views, transpose views, and shared tile allocation; it will lower using TMU, VPM read/write, VDW store, and VDR/VCD load where available. It will preserve 32-bit-only executable semantics in M5 while carrying strict future-proof precision/layout metadata. Unsupported patterns will be rejected deterministically rather than approximated.
