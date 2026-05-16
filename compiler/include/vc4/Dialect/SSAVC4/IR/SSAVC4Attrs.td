#ifndef SSAVC4_ATTRS_TD
#define SSAVC4_ATTRS_TD

include "mlir/IR/EnumAttr.td"
include "vc4/Dialect/SSAVC4/IR/SSAVC4Base.td"

class SSAVC4_DialectI32Enum<string name, string summary,
                            list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::ssavc4";
  let genSpecializedAttr = 0;
}

class SSAVC4_EnumAttr<EnumAttrInfo enumInfo, string name>
    : EnumAttr<SSAVC4_Dialect, enumInfo, name> {
  let cppNamespace = "::mlir::ssavc4";
  let assemblyFormat = "`<` $value `>`";
}

def SSAVC4_TMUUnit : SSAVC4_DialectI32Enum<"TMUUnit",
    "SSAVC4 TMU unit", [
      I32EnumAttrCase<"tmu0", 0>,
      I32EnumAttrCase<"tmu1", 1>
    ]>;

def SSAVC4_TMUMode : SSAVC4_DialectI32Enum<"TMUMode",
    "SSAVC4 TMU access mode", [
      I32EnumAttrCase<"direct", 0>
    ]>;

def SSAVC4_TMUReadPart : SSAVC4_DialectI32Enum<"TMUReadPart",
    "SSAVC4 TMU read part", [
      I32EnumAttrCase<"raw32", 0>
    ]>;

def SSAVC4_VPMOrientation : SSAVC4_DialectI32Enum<"VPMOrientation",
    "SSAVC4 VPM orientation", [
      I32EnumAttrCase<"horizontal", 0>,
      I32EnumAttrCase<"vertical", 1>
    ]>;

def SSAVC4_VPMLaneMode : SSAVC4_DialectI32Enum<"VPMLaneMode",
    "SSAVC4 VPM lane mode", [
      I32EnumAttrCase<"packed", 0>,
      I32EnumAttrCase<"laned", 1>
    ]>;

def SSAVC4_VPMElemWidth : SSAVC4_DialectI32Enum<"VPMElemWidth",
    "SSAVC4 VPM element width", [
      I32EnumAttrCase<"w8", 8>,
      I32EnumAttrCase<"w16", 16>,
      I32EnumAttrCase<"w32", 32>
    ]>;

def SSAVC4_VDWStoreSerialize : SSAVC4_DialectI32Enum<"VDWStoreSerialize",
    "SSAVC4 VDW store serialization mode", [
      I32EnumAttrCase<"none", 0>,
      I32EnumAttrCase<"mutex", 1>
    ]>;

def SSAVC4_FlagKind : SSAVC4_DialectI32Enum<"FlagKind",
    "SSAVC4 flag-producing operation kind", [
      I32EnumAttrCase<"sub", 0>,
      I32EnumAttrCase<"zero_test", 1>,
      I32EnumAttrCase<"compare", 2>
    ]>;

def SSAVC4_TMUUnitAttr : SSAVC4_EnumAttr<SSAVC4_TMUUnit, "tmu_unit">;
def SSAVC4_TMUModeAttr : SSAVC4_EnumAttr<SSAVC4_TMUMode, "tmu_mode">;
def SSAVC4_TMUReadPartAttr
    : SSAVC4_EnumAttr<SSAVC4_TMUReadPart, "tmu_read_part">;
def SSAVC4_VPMOrientationAttr
    : SSAVC4_EnumAttr<SSAVC4_VPMOrientation, "vpm_orientation">;
def SSAVC4_VPMLaneModeAttr
    : SSAVC4_EnumAttr<SSAVC4_VPMLaneMode, "vpm_lane_mode">;
def SSAVC4_VPMElemWidthAttr
    : SSAVC4_EnumAttr<SSAVC4_VPMElemWidth, "vpm_elem_width">;
def SSAVC4_VDWStoreSerializeAttr
    : SSAVC4_EnumAttr<SSAVC4_VDWStoreSerialize, "vdw_store_serialize">;
def SSAVC4_FlagKindAttr : SSAVC4_EnumAttr<SSAVC4_FlagKind, "flag_kind">;

#endif // SSAVC4_ATTRS_TD
