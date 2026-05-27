#ifndef VC4TILE_ATTRS_TD
#define VC4TILE_ATTRS_TD

include "mlir/IR/EnumAttr.td"
include "vc4/Dialect/VC4Tile/IR/VC4TileBase.td"

class VC4Tile_DialectI32Enum<string name, string summary,
                             list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::vc4tile";
  let genSpecializedAttr = 0;
}

class VC4Tile_EnumAttr<EnumAttrInfo enumInfo, string name>
    : EnumAttr<VC4Tile_Dialect, enumInfo, name> {
  let cppNamespace = "::mlir::vc4tile";
  let assemblyFormat = "`<` $value `>`";
}

def VC4Tile_ScheduleMode : VC4Tile_DialectI32Enum<"ScheduleMode",
    "VC4Tile kernel scheduling mode", [
      I32EnumAttrCase<"independent_vector", 0>,
      I32EnumAttrCase<"cooperative_block", 1>
    ]>;

def VC4Tile_OffsetUnit : VC4Tile_DialectI32Enum<"OffsetUnit",
    "VC4Tile global-memory offset unit", [
      I32EnumAttrCase<"byte", 0>,
      I32EnumAttrCase<"element", 1>
    ]>;

// Memory-space tags are part of the M5 semantic contract.  The C++ enum symbol
// for the textual `register` case intentionally avoids the C++ keyword.
def VC4Tile_MemorySpace : VC4Tile_DialectI32Enum<"MemorySpace",
    "VC4Tile memory-space domain", [
      I32EnumAttrCase<"global", 0>,
      I32EnumAttrCase<"shared_vpm", 1>,
      I32EnumAttrCase<"register_space", 2, "register">
    ]>;

def VC4Tile_MemoryAccess : VC4Tile_DialectI32Enum<"MemoryAccess",
    "VC4Tile global-memory access pattern", [
      I32EnumAttrCase<"coalesced", 0>,
      I32EnumAttrCase<"affine_contiguous", 1>,
      I32EnumAttrCase<"generic", 2>
    ]>;

def VC4Tile_Layout : VC4Tile_DialectI32Enum<"Layout",
    "VC4Tile ergonomic tile layout", [
      I32EnumAttrCase<"row_major", 0>,
      I32EnumAttrCase<"col_major", 1>,
      I32EnumAttrCase<"affine_2d", 2>,
      I32EnumAttrCase<"vpm_row", 3>,
      I32EnumAttrCase<"vpm_col", 4>,
      I32EnumAttrCase<"transposed_view", 5>
    ]>;

def VC4Tile_BoundaryPolicy : VC4Tile_DialectI32Enum<"BoundaryPolicy",
    "VC4Tile tile boundary policy", [
      I32EnumAttrCase<"exact", 0>,
      I32EnumAttrCase<"tail_predicated", 1>,
      I32EnumAttrCase<"zero", 2>,
      I32EnumAttrCase<"clamp", 3>
    ]>;

def VC4Tile_Role : VC4Tile_DialectI32Enum<"Role",
    "VC4Tile tile role metadata", [
      I32EnumAttrCase<"input", 0>,
      I32EnumAttrCase<"output", 1>,
      I32EnumAttrCase<"accumulator", 2>,
      I32EnumAttrCase<"scratch", 3>,
      I32EnumAttrCase<"metadata", 4>
    ]>;

def VC4Tile_Precision : VC4Tile_DialectI32Enum<"Precision",
    "VC4Tile precision policy marker", [
      I32EnumAttrCase<"exact_32", 0>
    ]>;

def VC4Tile_Packing : VC4Tile_DialectI32Enum<"Packing",
    "VC4Tile physical storage marker", [
      I32EnumAttrCase<"none", 0>
    ]>;

def VC4Tile_ReduceKind : VC4Tile_DialectI32Enum<"ReduceKind",
    "VC4Tile warp-local reduction kind", [
      I32EnumAttrCase<"add", 0>,
      I32EnumAttrCase<"min", 1>,
      I32EnumAttrCase<"max", 2>,
      I32EnumAttrCase<"bit_and", 3>,
      I32EnumAttrCase<"bit_or", 4>,
      I32EnumAttrCase<"bit_xor", 5>
    ]>;

def VC4Tile_VPMLayout : VC4Tile_DialectI32Enum<"VPMLayout",
    "VC4Tile shared VPM tile layout", [
      I32EnumAttrCase<"row_major", 0>,
      I32EnumAttrCase<"column_major", 1>
    ]>;

def VC4Tile_BarrierScope : VC4Tile_DialectI32Enum<"BarrierScope",
    "VC4Tile synchronization scope", [
      I32EnumAttrCase<"block", 0>
    ]>;

def VC4Tile_ScheduleModeAttr
    : VC4Tile_EnumAttr<VC4Tile_ScheduleMode, "schedule_mode">;
def VC4Tile_OffsetUnitAttr
    : VC4Tile_EnumAttr<VC4Tile_OffsetUnit, "offset_unit">;
def VC4Tile_MemorySpaceAttr
    : VC4Tile_EnumAttr<VC4Tile_MemorySpace, "memory_space">;
def VC4Tile_MemoryAccessAttr
    : VC4Tile_EnumAttr<VC4Tile_MemoryAccess, "memory_access">;
def VC4Tile_LayoutAttr
    : VC4Tile_EnumAttr<VC4Tile_Layout, "layout">;
def VC4Tile_BoundaryPolicyAttr
    : VC4Tile_EnumAttr<VC4Tile_BoundaryPolicy, "boundary_policy">;
def VC4Tile_RoleAttr
    : VC4Tile_EnumAttr<VC4Tile_Role, "role">;
def VC4Tile_PrecisionAttr
    : VC4Tile_EnumAttr<VC4Tile_Precision, "precision">;
def VC4Tile_PackingAttr
    : VC4Tile_EnumAttr<VC4Tile_Packing, "packing">;
def VC4Tile_ReduceKindAttr
    : VC4Tile_EnumAttr<VC4Tile_ReduceKind, "reduce_kind">;
def VC4Tile_VPMLayoutAttr
    : VC4Tile_EnumAttr<VC4Tile_VPMLayout, "vpm_layout">;
def VC4Tile_BarrierScopeAttr
    : VC4Tile_EnumAttr<VC4Tile_BarrierScope, "barrier_scope">;

#endif // VC4TILE_ATTRS_TD
