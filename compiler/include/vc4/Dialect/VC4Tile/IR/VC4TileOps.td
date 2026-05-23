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
    OptionalAttr<DictionaryAttr>:$resource
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
