#ifndef VC4_ENUMS_TD
#define VC4_ENUMS_TD

include "mlir/IR/EnumAttr.td"
include "vc4/Dialect/VC4/IR/VC4Base.td"

class VC4_I32Enum<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::vc4";
}

class VC4_EnumAttr<EnumAttrInfo enumInfo, string name>
    : EnumAttr<VC4_Dialect, enumInfo, name> {
  let cppNamespace = "::mlir::vc4";
}

def VC4_FunctionForm : VC4_I32Enum<"FunctionForm",
    "VC4 function form", [
      I32EnumAttrCase<"structured", 0>,
      I32EnumAttrCase<"scheduled", 1>
    ]>;
def VC4_FunctionFormAttr : VC4_EnumAttr<VC4_FunctionForm, "function_form">;

def VC4_ThreadingMode : VC4_I32Enum<"ThreadingMode",
    "VC4 function threading mode", [
      I32EnumAttrCase<"single", 0>,
      I32EnumAttrCase<"threadable", 1>
    ]>;
def VC4_ThreadingModeAttr
    : VC4_EnumAttr<VC4_ThreadingMode, "threading_mode">;

def VC4_BuiltinKind : VC4_I32Enum<"BuiltinKind",
    "VC4 builtin kind", [
      I32EnumAttrCase<"elem_num", 0>,
      I32EnumAttrCase<"qpu_num", 1>
    ]>;
def VC4_BuiltinKindAttr : VC4_EnumAttr<VC4_BuiltinKind, "builtin_kind">;

#endif // VC4_ENUMS_TD
