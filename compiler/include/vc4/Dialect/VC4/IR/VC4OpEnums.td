#ifndef VC4_OP_ENUMS_TD
#define VC4_OP_ENUMS_TD

include "vc4/Dialect/VC4/IR/VC4Enums.td"

def VC4_FunctionFormAttr : VC4_EnumAttr<VC4_FunctionForm, "function_form">;

def VC4_ThreadingModeAttr
    : VC4_EnumAttr<VC4_ThreadingMode, "threading_mode">;

def VC4_BuiltinKindAttr : VC4_EnumAttr<VC4_BuiltinKind, "builtin_kind">;

#endif // VC4_OP_ENUMS_TD
