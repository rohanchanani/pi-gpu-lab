#ifndef VC4TILE_OPS_TD
#define VC4TILE_OPS_TD

include "vc4/Dialect/VC4Tile/IR/VC4TileDialect.td"
include "vc4/Dialect/VC4Tile/IR/VC4TileTypes.td"
include "vc4/Dialect/VC4Tile/IR/VC4TileAttrs.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VC4Tile_Op<string mnemonic, list<Trait> traits = []>
    : Op<VC4Tile_Dialect, mnemonic, traits>;

class VC4Tile_PureOp<string mnemonic, list<Trait> traits = []>
    : VC4Tile_Op<mnemonic, [Pure] # traits>;

def VC4Tile_KernelOp : VC4Tile_Op<"kernel", [
    IsolatedFromAbove, Symbol
  ]> {
  let summary = "VC4Tile logical tile kernel";
  let description = [{
    Top-level VC4Tile kernel body.  The operation carries the schedule/resource
    contract that later `vc4tile -> ssavc4` lowering must translate into the
    existing `vc4.launch_abi` and `vc4.resource` dictionaries.  It is not an
    artifact sink and must never lower directly to scheduled `vc4`.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$public_name,
    OptionalAttr<VC4Tile_ScheduleModeAttr>:$schedule_mode,
    OptionalAttr<I32Attr>:$warps_per_block_max,
    OptionalAttr<I32Attr>:$vpm_rows_per_block,
    OptionalAttr<I32Attr>:$vpm_bytes_per_block,
    OptionalAttr<I32Attr>:$semaphores_per_block,
    OptionalAttr<BoolAttr>:$uses_shared_vpm,
    OptionalAttr<BoolAttr>:$uses_barrier,
    OptionalAttr<BoolAttr>:$require_full_block_residency,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictionaryAttr>:$launch_abi,
    OptionalAttr<DictionaryAttr>:$resource,
    OptionalAttr<DictionaryAttr>:$resource_intent
  );
  let regions = (region AnyRegion:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def VC4Tile_ReturnOp : VC4Tile_Op<"return", [Terminator]> {
  let summary = "VC4Tile kernel terminator";
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def VC4Tile_ProgramIdOp : VC4Tile_PureOp<"program_id"> {
  let summary = "logical independent-vector program/request id";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_BlockIdOp : VC4Tile_PureOp<"block_id"> {
  let summary = "logical cooperative block id";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_WarpIdOp : VC4Tile_PureOp<"warp_id"> {
  let summary = "logical warp id within a cooperative block";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_LaneIdOp : VC4Tile_PureOp<"lane_id"> {
  let summary = "scalar lane id";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_LaneRangeOp : VC4Tile_PureOp<"lane_range"> {
  let summary = "16-lane id vector [0, 1, ..., 15]";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_ThreadIdOp : VC4Tile_PureOp<"thread_id"> {
  let summary = "logical thread id vector within the block/tile";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_MaskAllOp : VC4Tile_PureOp<"mask_all"> {
  let summary = "all-lanes-active mask";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_TailMaskOp : VC4Tile_PureOp<"tail_mask"> {
  let summary = "contiguous active-lane mask for a base/limit range";
  let arguments = (ins AnyType:$base, AnyType:$limit);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$base `,` $limit attr-dict `:` type($base) `,` type($limit) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_TileRectMaskOp : VC4Tile_PureOp<"tile_rect_mask"> {
  let summary = "static rectangular active-lane mask for a 4x4 tile";
  let arguments = (ins
    I32Attr:$active_rows,
    I32Attr:$active_cols,
    ArrayAttr:$shape,
    VC4Tile_LayoutAttr:$layout
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_TileBoundsMaskOp : VC4Tile_PureOp<"tile_bounds_mask"> {
  let summary = "dynamic rectangular in-bounds predicate for a 4x4 tile";
  let description = [{
    CuTe-style rectangular tile predicate.  The active row/column counts are
    dynamic scalar values already clamped by the producer to the physical tile
    shape.  VC4Tile lowering maps this semantic predicate to concrete legal
    hardware transfers such as full VDW blocks or guarded one-row tail stores.
  }];
  let arguments = (ins
    AnyType:$active_rows,
    AnyType:$active_cols,
    ArrayAttr:$shape,
    VC4Tile_LayoutAttr:$layout
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$active_rows `,` $active_cols attr-dict `:` type($active_rows) `,` type($active_cols) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_MaskAndOp : VC4Tile_PureOp<"mask_and"> {
  let summary = "logical conjunction of two tile predicates";
  let description = [{
    CuTe-style predicate composition.  This op is a semantic tile predicate:
    it combines logical masks at the VC4Tile layer and lowers into concrete
    active-lane values only at its final memory or compute consumer.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_MaskOrOp : VC4Tile_PureOp<"mask_or"> {
  let summary = "logical disjunction of two tile predicates";
  let description = [{
    CuTe-style predicate composition.  This op is a semantic tile predicate:
    it combines logical masks at the VC4Tile layer and lowers into concrete
    active-lane values only at its final memory or compute consumer.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_MaskNotOp : VC4Tile_PureOp<"mask_not"> {
  let summary = "logical complement of a tile predicate";
  let description = [{
    CuTe-style predicate composition.  This op is a semantic tile predicate:
    it complements a mask over the fixed 16-lane tile domain and lowers into
    concrete active-lane values only at its final memory or compute consumer.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_MaskedLoadGlobalOp : VC4Tile_Op<"masked_load_global", [
    MemoryEffects<[MemRead]>
  ]> {
  let summary = "masked 16-lane global memory load intent";
  let description = [{
    Tile-level global load.  Later lowering maps the supported 32-bit direct
    load subset to SSAVC4 TMU request/read operations, never directly to
    scheduled `vc4`.
  }];
  let arguments = (ins
    AnyType:$base,
    AnyType:$offsets,
    AnyType:$mask,
    I32Attr:$elem_bytes,
    VC4Tile_OffsetUnitAttr:$offset_unit,
    OptionalAttr<VC4Tile_MemorySpaceAttr>:$memory_space,
    OptionalAttr<VC4Tile_MemoryAccessAttr>:$access
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$base `,` $offsets `,` $mask attr-dict `:` type($base) `,` type($offsets) `,` type($mask) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_MaskedStoreGlobalOp : VC4Tile_Op<"masked_store_global", [
    MemoryEffects<[MemWrite]>
  ]> {
  let summary = "masked 16-lane global memory store intent";
  let description = [{
    Tile-level global store.  Later lowering maps the supported coalesced or
    affine-contiguous 32-bit store subset through SSAVC4 VPM/VDW staging, never
    directly to scheduled `vc4`.
  }];
  let arguments = (ins
    AnyType:$base,
    AnyType:$offsets,
    AnyType:$value,
    AnyType:$mask,
    I32Attr:$elem_bytes,
    VC4Tile_OffsetUnitAttr:$offset_unit,
    OptionalAttr<VC4Tile_MemorySpaceAttr>:$memory_space,
    OptionalAttr<VC4Tile_MemoryAccessAttr>:$access
  );
  let assemblyFormat = "$base `,` $offsets `,` $value `,` $mask attr-dict `:` type($base) `,` type($offsets) `,` type($value) `,` type($mask)";
  let hasVerifier = 1;
}


def VC4Tile_TileDescriptorOp : VC4Tile_PureOp<"tile_descriptor"> {
  let summary = "metadata-only ergonomic VC4Tile tile descriptor";
  let description = [{
    Defines the M5 tile/layout/role/precision vocabulary without adding an
    executable tile movement path.  The op is legal only as metadata in the
    ergonomic surface; later copy-planning slices will consume the same fields
    on tile movement and compute operations.  M5 executable semantics remain
    32-bit only.
  }];
  let arguments = (ins
    ArrayAttr:$shape,
    I32Attr:$rank,
    TypeAttr:$element_type,
    TypeAttr:$storage_type,
    TypeAttr:$expressed_type,
    TypeAttr:$accumulator_type,
    VC4Tile_MemorySpaceAttr:$memory_space,
    VC4Tile_LayoutAttr:$layout,
    VC4Tile_RoleAttr:$role,
    VC4Tile_PrecisionAttr:$precision,
    VC4Tile_PackingAttr:$packing,
    OptionalAttr<VC4Tile_BoundaryPolicyAttr>:$boundary,
    OptionalAttr<VC4Tile_CopyStageAttr>:$copy_stage,
    OptionalAttr<VC4Tile_ReuseHintAttr>:$reuse_hint,
    OptionalAttr<VC4Tile_OffsetUnitAttr>:$offset_unit,
    OptionalAttr<ArrayAttr>:$strides
  );
  let results = (outs AnyType:$tile);
  let assemblyFormat = "attr-dict `:` type($tile)";
  let hasVerifier = 1;
}


// M5 ergonomic tile movement surface operations.  They are consumed by
// --plan-vc4tile-copies and must not survive into verified VC4Tile core.
def VC4Tile_TileLoadOp : VC4Tile_Op<"tile_load", [
    MemoryEffects<[MemRead]>
  ]> {
  let summary = "ergonomic tile load surface operation";
  let description = [{
    Loads a 32-bit tile fragment from a declared memory space into a register
    carrier.  M5 v1 supports the global->register 1x16 case in the copy planner;
    other parsed forms are accepted only when they are planned or rejected
    deterministically before SSAVC4 lowering.
  }];
  let arguments = (ins
    AnyType:$base,
    AnyType:$offset,
    AnyType:$mask,
    ArrayAttr:$shape,
    VC4Tile_LayoutAttr:$layout,
    VC4Tile_MemorySpaceAttr:$memory_space,
    TypeAttr:$element_type,
    TypeAttr:$storage_type,
    VC4Tile_PrecisionAttr:$precision,
    OptionalAttr<TypeAttr>:$expressed_type,
    OptionalAttr<TypeAttr>:$accumulator_type,
    OptionalAttr<VC4Tile_RoleAttr>:$role,
    OptionalAttr<VC4Tile_BoundaryPolicyAttr>:$boundary,
    OptionalAttr<VC4Tile_CopyStageAttr>:$copy_stage,
    OptionalAttr<VC4Tile_ReuseHintAttr>:$reuse_hint,
    OptionalAttr<VC4Tile_PackingAttr>:$packing,
    OptionalAttr<VC4Tile_OffsetUnitAttr>:$offset_unit
  );
  let results = (outs AnyType:$tile);
  let hasVerifier = 1;
}

def VC4Tile_TileStoreOp : VC4Tile_Op<"tile_store", [
    MemoryEffects<[MemWrite]>
  ]> {
  let summary = "ergonomic tile store surface operation";
  let description = [{
    Stores a 32-bit tile fragment to a declared memory space.  M5 v1 plans the
    register->global 1x16 case through the existing masked global-store core
    contract, which later lowers through VPM/VDW rather than a direct hardware
    register-to-global store.
  }];
  let arguments = (ins
    AnyType:$tile,
    AnyType:$base,
    AnyType:$offset,
    AnyType:$mask,
    ArrayAttr:$shape,
    VC4Tile_LayoutAttr:$layout,
    VC4Tile_MemorySpaceAttr:$memory_space,
    TypeAttr:$element_type,
    TypeAttr:$storage_type,
    VC4Tile_PrecisionAttr:$precision,
    OptionalAttr<TypeAttr>:$expressed_type,
    OptionalAttr<TypeAttr>:$accumulator_type,
    OptionalAttr<VC4Tile_RoleAttr>:$role,
    OptionalAttr<VC4Tile_BoundaryPolicyAttr>:$boundary,
    OptionalAttr<VC4Tile_CopyStageAttr>:$copy_stage,
    OptionalAttr<VC4Tile_ReuseHintAttr>:$reuse_hint,
    OptionalAttr<VC4Tile_PackingAttr>:$packing,
    OptionalAttr<VC4Tile_OffsetUnitAttr>:$offset_unit
  );
  let hasVerifier = 1;
}

def VC4Tile_CopyTileOp : VC4Tile_Op<"copy_tile", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "ergonomic copy-tile surface operation";
  let description = [{
    Surface copy-planning operation.  The single variadic operand/result group
    keeps the first implementation flexible enough for register->shared and
    shared->register forms while requiring the planner to erase the operation
    before verified VC4Tile core.
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    ArrayAttr:$shape,
    VC4Tile_MemorySpaceAttr:$src_space,
    VC4Tile_MemorySpaceAttr:$dst_space,
    VC4Tile_LayoutAttr:$src_layout,
    VC4Tile_LayoutAttr:$dst_layout,
    TypeAttr:$element_type,
    TypeAttr:$storage_type,
    VC4Tile_PrecisionAttr:$precision,
    OptionalAttr<TypeAttr>:$expressed_type,
    OptionalAttr<TypeAttr>:$accumulator_type,
    OptionalAttr<VC4Tile_RoleAttr>:$role,
    OptionalAttr<VC4Tile_BoundaryPolicyAttr>:$boundary,
    OptionalAttr<VC4Tile_CopyStageAttr>:$copy_stage,
    OptionalAttr<VC4Tile_ReuseHintAttr>:$reuse_hint,
    OptionalAttr<VC4Tile_PackingAttr>:$packing,
    OptionalAttr<I32Attr>:$elem_bytes
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
}

def VC4Tile_TileViewOp : VC4Tile_PureOp<"tile_view"> {
  let summary = "symbolic tile view surface operation";
  let arguments = (ins
    AnyType:$source,
    OptionalAttr<ArrayAttr>:$offsets,
    OptionalAttr<ArrayAttr>:$sizes,
    OptionalAttr<ArrayAttr>:$strides,
    OptionalAttr<VC4Tile_LayoutAttr>:$layout
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def VC4Tile_TileSubviewOp : VC4Tile_PureOp<"tile_subview"> {
  let summary = "symbolic tile subview surface operation";
  let arguments = (ins
    AnyType:$source,
    ArrayAttr:$offsets,
    ArrayAttr:$sizes,
    OptionalAttr<ArrayAttr>:$strides,
    OptionalAttr<VC4Tile_LayoutAttr>:$layout
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def VC4Tile_TransposeViewOp : VC4Tile_PureOp<"transpose_view"> {
  let summary = "symbolic transposed tile view surface operation";
  let arguments = (ins AnyType:$source, ArrayAttr:$permutation);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def VC4Tile_SharedTileAllocOp : VC4Tile_Op<"shared_tile_alloc", [
    MemoryEffects<[MemAlloc]>
  ]> {
  let summary = "ergonomic shared VPM tile allocation surface operation";
  let arguments = (ins
    I32Attr:$rows,
    I32Attr:$elem_bytes,
    ArrayAttr:$shape,
    TypeAttr:$element_type,
    TypeAttr:$storage_type,
    VC4Tile_LayoutAttr:$layout,
    OptionalAttr<VC4Tile_RoleAttr>:$role,
    OptionalAttr<VC4Tile_CopyStageAttr>:$copy_stage,
    OptionalAttr<VC4Tile_ReuseHintAttr>:$reuse_hint,
    OptionalAttr<VC4Tile_PrecisionAttr>:$precision,
    OptionalAttr<VC4Tile_PackingAttr>:$packing,
    OptionalAttr<VC4Tile_MemorySpaceAttr>:$memory_space
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}


// M5 ergonomic tile elementwise compute surface operations. They canonicalize
// to VC4Tile core arith/vector operations during --canonicalize-vc4tile-surface
// and must not survive into verified VC4Tile core or SSAVC4 lowering.
def VC4Tile_TileFillOp : VC4Tile_PureOp<"tile_fill"> {
  let summary = "ergonomic constant tile fill surface operation";
  let arguments = (ins
    AnyAttr:$value,
    ArrayAttr:$shape,
    VC4Tile_LayoutAttr:$layout,
    VC4Tile_MemorySpaceAttr:$memory_space,
    TypeAttr:$element_type,
    TypeAttr:$storage_type,
    VC4Tile_PrecisionAttr:$precision,
    OptionalAttr<TypeAttr>:$expressed_type,
    OptionalAttr<TypeAttr>:$accumulator_type,
    OptionalAttr<VC4Tile_RoleAttr>:$role,
    OptionalAttr<VC4Tile_BoundaryPolicyAttr>:$boundary,
    OptionalAttr<VC4Tile_CopyStageAttr>:$copy_stage,
    OptionalAttr<VC4Tile_ReuseHintAttr>:$reuse_hint,
    OptionalAttr<VC4Tile_PackingAttr>:$packing
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

// Broadcasts one scalar 32-bit value into a 16-lane register tile.
def VC4Tile_TileBroadcastOp : VC4Tile_PureOp<"tile_broadcast"> {
  let summary = "ergonomic scalar-to-tile broadcast surface operation";
  let arguments = (ins
    AnyType:$source,
    ArrayAttr:$shape,
    VC4Tile_LayoutAttr:$layout,
    VC4Tile_MemorySpaceAttr:$memory_space,
    TypeAttr:$element_type,
    TypeAttr:$storage_type,
    VC4Tile_PrecisionAttr:$precision,
    OptionalAttr<TypeAttr>:$expressed_type,
    OptionalAttr<TypeAttr>:$accumulator_type,
    OptionalAttr<VC4Tile_RoleAttr>:$role,
    OptionalAttr<VC4Tile_BoundaryPolicyAttr>:$boundary,
    OptionalAttr<VC4Tile_CopyStageAttr>:$copy_stage,
    OptionalAttr<VC4Tile_ReuseHintAttr>:$reuse_hint,
    OptionalAttr<VC4Tile_PackingAttr>:$packing
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

class VC4Tile_TileBinaryComputeOp<string mnemonic, string summaryText>
    : VC4Tile_PureOp<mnemonic> {
  let summary = summaryText;
  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    ArrayAttr:$shape,
    VC4Tile_LayoutAttr:$layout,
    VC4Tile_MemorySpaceAttr:$memory_space,
    TypeAttr:$element_type,
    TypeAttr:$storage_type,
    VC4Tile_PrecisionAttr:$precision,
    OptionalAttr<TypeAttr>:$expressed_type,
    OptionalAttr<TypeAttr>:$accumulator_type,
    OptionalAttr<VC4Tile_RoleAttr>:$role,
    OptionalAttr<VC4Tile_BoundaryPolicyAttr>:$boundary,
    OptionalAttr<VC4Tile_CopyStageAttr>:$copy_stage,
    OptionalAttr<VC4Tile_ReuseHintAttr>:$reuse_hint,
    OptionalAttr<VC4Tile_PackingAttr>:$packing
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def VC4Tile_TileAddOp : VC4Tile_TileBinaryComputeOp<"tile_add", "ergonomic elementwise tile add surface operation">;
def VC4Tile_TileSubOp : VC4Tile_TileBinaryComputeOp<"tile_sub", "ergonomic elementwise tile subtract surface operation">;
def VC4Tile_TileMulOp : VC4Tile_TileBinaryComputeOp<"tile_mul", "ergonomic elementwise tile multiply surface operation">;

def VC4Tile_TileSelectOp : VC4Tile_PureOp<"tile_select"> {
  let summary = "ergonomic masked tile select surface operation";
  let arguments = (ins
    AnyType:$mask,
    AnyType:$true_value,
    AnyType:$false_value,
    ArrayAttr:$shape,
    VC4Tile_LayoutAttr:$layout,
    VC4Tile_MemorySpaceAttr:$memory_space,
    TypeAttr:$element_type,
    TypeAttr:$storage_type,
    VC4Tile_PrecisionAttr:$precision,
    OptionalAttr<TypeAttr>:$expressed_type,
    OptionalAttr<TypeAttr>:$accumulator_type,
    OptionalAttr<VC4Tile_RoleAttr>:$role,
    OptionalAttr<VC4Tile_BoundaryPolicyAttr>:$boundary,
    OptionalAttr<VC4Tile_CopyStageAttr>:$copy_stage,
    OptionalAttr<VC4Tile_ReuseHintAttr>:$reuse_hint,
    OptionalAttr<VC4Tile_PackingAttr>:$packing
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}


class VC4Tile_TileReductionSurfaceOp<string mnemonic, string summaryText,
                                     list<Trait> traits = [Pure]>
    : VC4Tile_Op<mnemonic, traits> {
  let summary = summaryText;
  let arguments = (ins
    AnyType:$input,
    AnyType:$mask,
    VC4Tile_ReduceKindAttr:$kind,
    ArrayAttr:$shape,
    VC4Tile_LayoutAttr:$layout,
    VC4Tile_MemorySpaceAttr:$memory_space,
    TypeAttr:$element_type,
    TypeAttr:$storage_type,
    VC4Tile_PrecisionAttr:$precision,
    OptionalAttr<I32Attr>:$axis,
    OptionalAttr<StrAttr>:$algorithm_hint,
    OptionalAttr<TypeAttr>:$expressed_type,
    OptionalAttr<TypeAttr>:$accumulator_type,
    OptionalAttr<VC4Tile_RoleAttr>:$role,
    OptionalAttr<VC4Tile_BoundaryPolicyAttr>:$boundary,
    OptionalAttr<VC4Tile_CopyStageAttr>:$copy_stage,
    OptionalAttr<VC4Tile_ReuseHintAttr>:$reuse_hint,
    OptionalAttr<VC4Tile_PackingAttr>:$packing
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def VC4Tile_TileReduceOp
    : VC4Tile_TileReductionSurfaceOp<"tile_reduce",
        "ergonomic tile-local reduction surface operation">;

def VC4Tile_RowReduceOp
    : VC4Tile_TileReductionSurfaceOp<"row_reduce",
        "ergonomic row/tile reduction surface operation">;

def VC4Tile_WarpReduceOp
    : VC4Tile_TileReductionSurfaceOp<"warp_reduce",
        "ergonomic warp-local reduction surface operation">;

def VC4Tile_BlockReduceOp
    : VC4Tile_TileReductionSurfaceOp<"block_reduce",
        "ergonomic cooperative-block reduction surface operation",
        [MemoryEffects<[MemRead, MemWrite]>]>;

// M5 ergonomic tile contraction surface operations. They are producer-facing
// contracts and must canonicalize into tile elementwise/reduction/core ops
// before VC4Tile core verification or SSAVC4 lowering.
def VC4Tile_TileDotOp : VC4Tile_PureOp<"tile_dot"> {
  let summary = "ergonomic 1x16 tile dot-product surface operation";
  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$mask,
    I32Attr:$k,
    ArrayAttr:$shape,
    VC4Tile_LayoutAttr:$layout,
    VC4Tile_MemorySpaceAttr:$memory_space,
    TypeAttr:$element_type,
    TypeAttr:$storage_type,
    TypeAttr:$accumulator_type,
    VC4Tile_PrecisionAttr:$precision,
    OptionalAttr<VC4Tile_RoleAttr>:$lhs_role,
    OptionalAttr<VC4Tile_RoleAttr>:$rhs_role,
    OptionalAttr<VC4Tile_RoleAttr>:$acc_role,
    OptionalAttr<VC4Tile_BoundaryPolicyAttr>:$boundary,
    OptionalAttr<VC4Tile_CopyStageAttr>:$copy_stage,
    OptionalAttr<VC4Tile_ReuseHintAttr>:$reuse_hint,
    OptionalAttr<VC4Tile_PackingAttr>:$packing,
    OptionalAttr<I32Attr>:$active_k,
    OptionalAttr<StrAttr>:$algorithm_hint
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

class VC4Tile_TileContractionSurfaceOp<string mnemonic, string summaryText>
    : VC4Tile_PureOp<mnemonic> {
  let summary = summaryText;
  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$acc,
    AnyType:$mask,
    I32Attr:$m,
    I32Attr:$n,
    I32Attr:$k,
    ArrayAttr:$shape,
    VC4Tile_LayoutAttr:$lhs_layout,
    VC4Tile_LayoutAttr:$rhs_layout,
    VC4Tile_LayoutAttr:$acc_layout,
    VC4Tile_MemorySpaceAttr:$memory_space,
    TypeAttr:$element_type,
    TypeAttr:$storage_type,
    TypeAttr:$accumulator_type,
    VC4Tile_PrecisionAttr:$precision,
    OptionalAttr<ArrayAttr>:$contracting_dims,
    OptionalAttr<ArrayAttr>:$iterator_types,
    OptionalAttr<VC4Tile_RoleAttr>:$lhs_role,
    OptionalAttr<VC4Tile_RoleAttr>:$rhs_role,
    OptionalAttr<VC4Tile_RoleAttr>:$acc_role,
    OptionalAttr<VC4Tile_BoundaryPolicyAttr>:$boundary,
    OptionalAttr<VC4Tile_CopyStageAttr>:$copy_stage,
    OptionalAttr<VC4Tile_ReuseHintAttr>:$reuse_hint,
    OptionalAttr<VC4Tile_PackingAttr>:$packing,
    OptionalAttr<I32Attr>:$active_k,
    OptionalAttr<StrAttr>:$algorithm_hint
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def VC4Tile_TileContractOp
    : VC4Tile_TileContractionSurfaceOp<"tile_contract",
        "ergonomic tile contraction surface operation">;

def VC4Tile_TileMatmulOp
    : VC4Tile_TileContractionSurfaceOp<"tile_matmul",
        "ergonomic tile matrix-multiply contraction surface operation">;

// Temporary M5 surface/core boundary sentinel.  This operation is deliberately
// non-executable: surface canonicalization erases it, core verification rejects
// it if it survives, and VC4Tile-to-SSAVC4 conversion never lowers it.
def VC4Tile_SurfacePlaceholderOp : VC4Tile_Op<"surface_placeholder"> {
  let summary = "temporary non-executable VC4Tile surface sentinel";
  let description = [{
    Boundary-only surface operation used by M5 slice 01 to prove that the
    ergonomic surface pipeline is ordered before VC4Tile core verification and
    SSAVC4 lowering.  Later M5 slices add real surface movement operations; this
    placeholder must not lower to SSAVC4, scheduled VC4, QASM, or runtime
    artifacts.
  }];
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def VC4Tile_RotateOp : VC4Tile_PureOp<"rotate"> {
  let summary = "warp-local vector lane rotate";
  let arguments = (ins AnyType:$input, I32Attr:$amount);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_ReduceOp : VC4Tile_PureOp<"reduce"> {
  let summary = "warp-local masked reduction";
  let arguments = (ins
    AnyType:$input,
    AnyType:$mask,
    VC4Tile_ReduceKindAttr:$kind
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_SharedAllocOp : VC4Tile_Op<"shared_alloc", [
    MemoryEffects<[MemAlloc]>
  ]> {
  let summary = "row-granular shared VPM tile allocation intent";
  let arguments = (ins
    I32Attr:$rows,
    I32Attr:$elem_bytes,
    OptionalAttr<VC4Tile_MemorySpaceAttr>:$memory_space,
    OptionalAttr<VC4Tile_VPMLayoutAttr>:$layout
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_SharedLoadOp : VC4Tile_Op<"shared_load", [
    MemoryEffects<[MemRead]>
  ]> {
  let summary = "masked row-granular shared VPM tile load";
  let arguments = (ins
    AnyType:$handle,
    AnyType:$row,
    AnyType:$mask,
    I32Attr:$elem_bytes,
    OptionalAttr<VC4Tile_MemorySpaceAttr>:$memory_space,
    OptionalAttr<VC4Tile_VPMLayoutAttr>:$layout
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$handle `,` $row `,` $mask attr-dict `:` type($handle) `,` type($row) `,` type($mask) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Tile_SharedStoreOp : VC4Tile_Op<"shared_store", [
    MemoryEffects<[MemWrite]>
  ]> {
  let summary = "masked row-granular shared VPM tile store";
  let arguments = (ins
    AnyType:$handle,
    AnyType:$row,
    AnyType:$value,
    AnyType:$mask,
    I32Attr:$elem_bytes,
    OptionalAttr<VC4Tile_MemorySpaceAttr>:$memory_space,
    OptionalAttr<VC4Tile_VPMLayoutAttr>:$layout
  );
  let assemblyFormat = "$handle `,` $row `,` $value `,` $mask attr-dict `:` type($handle) `,` type($row) `,` type($value) `,` type($mask)";
  let hasVerifier = 1;
}

def VC4Tile_SharedStoreGlobalOp : VC4Tile_Op<"shared_store_global", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "VDW global store from existing shared VPM coordinates";
  let arguments = (ins
    AnyType:$handle,
    AnyType:$vpm_y,
    AnyType:$vpm_x,
    AnyType:$base,
    AnyType:$offset,
    AnyType:$mask,
    I32Attr:$elem_bytes,
    I32Attr:$row_len,
    I32Attr:$nrows,
    I32Attr:$memory_pitch_bytes,
    OptionalAttr<VC4Tile_VPMLayoutAttr>:$layout
  );
  let assemblyFormat = "$handle `,` $vpm_y `,` $vpm_x `,` $base `,` $offset `,` $mask attr-dict `:` type($handle) `,` type($vpm_y) `,` type($vpm_x) `,` type($base) `,` type($offset) `,` type($mask)";
  let hasVerifier = 1;
}



def VC4Tile_VDRLoadTileOp : VC4Tile_Op<"vdr_load_tile", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "lowering-ready VC4Tile global-to-shared VDR tile load";
  let description = [{
    Core VC4Tile copy-planner atom for a regular 32-bit global-to-shared VPM
    tile load.  It is produced by --plan-vc4tile-copies from ergonomic
    global-to-shared copy forms and lowers through the existing SSAVC4 VDR path;
    it is not a surface operation and must remain on the VC4Tile-to-SSAVC4 path.
  }];
  let arguments = (ins
    AnyType:$address,
    AnyType:$handle,
    I32Attr:$elem_bytes,
    I32Attr:$row_len,
    I32Attr:$nrows,
    I32Attr:$memory_pitch_bytes,
    I32Attr:$vpm_base_row,
    I32Attr:$vpm_base_col,
    OptionalAttr<VC4Tile_LayoutAttr>:$layout,
    I32Attr:$vpitch,
    OptionalAttr<StrAttr>:$serialize
  );
  let assemblyFormat = "$address `,` $handle attr-dict `:` type($address) `,` type($handle)";
  let hasVerifier = 1;
}

def VC4Tile_BarrierOp : VC4Tile_Op<"barrier", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "cooperative-block barrier intent";
  let description = [{
    Block-wide cooperative barrier.  The verifier requires a cooperative kernel
    resource contract.  Later lowering must use SSAVC4 barrier/semaphore ops and
    the validated reusable four-semaphore protocol.
  }];
  let arguments = (ins OptionalAttr<VC4Tile_BarrierScopeAttr>:$scope);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // VC4TILE_OPS_TD
