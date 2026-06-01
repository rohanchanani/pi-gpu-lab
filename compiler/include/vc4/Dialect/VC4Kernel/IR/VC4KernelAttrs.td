#ifndef VC4KERNEL_ATTRS_TD
#define VC4KERNEL_ATTRS_TD

include "mlir/IR/EnumAttr.td"
include "vc4/Dialect/VC4Kernel/IR/VC4KernelDialect.td"

class VC4Kernel_DialectI32Enum<string name, string summary,
                               list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::vc4kernel";
  let genSpecializedAttr = 0;
}

class VC4Kernel_EnumAttr<EnumAttrInfo enumInfo, string name>
    : EnumAttr<VC4Kernel_Dialect, enumInfo, name> {
  let cppNamespace = "::mlir::vc4kernel";
  let assemblyFormat = "`<` $value `>`";
}

def VC4Kernel_ScheduleMode : VC4Kernel_DialectI32Enum<"ScheduleMode",
    "VC4Kernel scheduling mode", [
      I32EnumAttrCase<"independent_vector", 0>,
      I32EnumAttrCase<"cooperative_block", 1>
    ]>;

def VC4Kernel_ReduceKind : VC4Kernel_DialectI32Enum<"ReduceKind",
    "VC4Kernel fragment reduction kind", [
      I32EnumAttrCase<"add", 0>
    ]>;

def VC4Kernel_CmpPredicate : VC4Kernel_DialectI32Enum<"CmpPredicate",
    "VC4Kernel fragment compare predicate", [
      I32EnumAttrCase<"eq", 0>,
      I32EnumAttrCase<"ne", 1>,
      I32EnumAttrCase<"ult", 2>,
      I32EnumAttrCase<"ule", 3>,
      I32EnumAttrCase<"ugt", 4>,
      I32EnumAttrCase<"uge", 5>
    ]>;

def VC4Kernel_VPMOrientation : VC4Kernel_DialectI32Enum<"VPMOrientation",
    "VC4Kernel VPM fragment orientation", [
      I32EnumAttrCase<"horizontal", 0>,
      I32EnumAttrCase<"vertical", 1>
    ]>;

def VC4Kernel_VPMWidth : VC4Kernel_DialectI32Enum<"VPMWidth",
    "VC4Kernel VPM element width", [
      I32EnumAttrCase<"w32", 0>,
      I32EnumAttrCase<"w16", 1>,
      I32EnumAttrCase<"w8", 2>
    ]>;

def VC4Kernel_VPMSubword : VC4Kernel_DialectI32Enum<"VPMSubword",
    "VC4Kernel VPM subword packing mode", [
      I32EnumAttrCase<"none", 0>,
      I32EnumAttrCase<"packed", 1>,
      I32EnumAttrCase<"laned", 2>
    ]>;

def VC4Kernel_ScheduleModeAttr
    : VC4Kernel_EnumAttr<VC4Kernel_ScheduleMode, "schedule_mode">;
def VC4Kernel_ReduceKindAttr
    : VC4Kernel_EnumAttr<VC4Kernel_ReduceKind, "reduce">;
def VC4Kernel_CmpPredicateAttr
    : VC4Kernel_EnumAttr<VC4Kernel_CmpPredicate, "cmp">;
def VC4Kernel_VPMOrientationAttr
    : VC4Kernel_EnumAttr<VC4Kernel_VPMOrientation, "vpm_orientation">;
def VC4Kernel_VPMWidthAttr
    : VC4Kernel_EnumAttr<VC4Kernel_VPMWidth, "vpm_width">;
def VC4Kernel_VPMSubwordAttr
    : VC4Kernel_EnumAttr<VC4Kernel_VPMSubword, "vpm_subword">;

#endif // VC4KERNEL_ATTRS_TD
