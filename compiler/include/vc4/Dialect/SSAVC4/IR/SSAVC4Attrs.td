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

def SSAVC4_VPMSubword : SSAVC4_DialectI32Enum<"VPMSubword",
    "SSAVC4 VPM subword mode", [
      I32EnumAttrCase<"none", 0>,
      I32EnumAttrCase<"packed", 1>,
      I32EnumAttrCase<"laned", 2>
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
      I32EnumAttrCase<"compare", 2>,
      I32EnumAttrCase<"fsub", 3>
    ]>;

def SSAVC4_SFUKind : SSAVC4_DialectI32Enum<"SFUKind",
    "SSAVC4 hardware special-functions unit kind", [
      I32EnumAttrCase<"recip", 52>,
      I32EnumAttrCase<"rsqrt", 53>,
      I32EnumAttrCase<"exp", 54>,
      I32EnumAttrCase<"log", 55>
    ]>;

def SSAVC4_FPMathPolicy : SSAVC4_DialectI32Enum<"FPMathPolicy",
    "SSAVC4 explicit approximate floating-point math policy", [
      I32EnumAttrCase<"approx_sfu", 0>,
      I32EnumAttrCase<"exact", 1>
    ]>;

def SSAVC4_FPDomain : SSAVC4_DialectI32Enum<"FPDomain",
    "SSAVC4 explicit finite floating-point domain contract", [
      I32EnumAttrCase<"finite", 0>,
      I32EnumAttrCase<"finite_nonzero", 1>,
      I32EnumAttrCase<"finite_positive", 2>
    ]>;

def SSAVC4_TMUUnitAttr : SSAVC4_EnumAttr<SSAVC4_TMUUnit, "tmu_unit">;
def SSAVC4_TMUModeAttr : SSAVC4_EnumAttr<SSAVC4_TMUMode, "tmu_mode">;
def SSAVC4_TMUReadPartAttr
    : SSAVC4_EnumAttr<SSAVC4_TMUReadPart, "tmu_read_part">;
def SSAVC4_VPMOrientationAttr
    : SSAVC4_EnumAttr<SSAVC4_VPMOrientation, "vpm_orientation">;
def SSAVC4_VPMSubwordAttr
    : SSAVC4_EnumAttr<SSAVC4_VPMSubword, "vpm_subword">;
def SSAVC4_VPMElemWidthAttr
    : SSAVC4_EnumAttr<SSAVC4_VPMElemWidth, "vpm_elem_width">;
def SSAVC4_VDWStoreSerializeAttr
    : SSAVC4_EnumAttr<SSAVC4_VDWStoreSerialize, "vdw_store_serialize">;
def SSAVC4_FlagKindAttr : SSAVC4_EnumAttr<SSAVC4_FlagKind, "flag_kind">;
def SSAVC4_SFUKindAttr : SSAVC4_EnumAttr<SSAVC4_SFUKind, "sfu_kind">;
def SSAVC4_FPMathPolicyAttr
    : SSAVC4_EnumAttr<SSAVC4_FPMathPolicy, "fp_math_policy">;
def SSAVC4_FPDomainAttr : SSAVC4_EnumAttr<SSAVC4_FPDomain, "fp_domain">;

#endif // SSAVC4_ATTRS_TD
