#ifndef VC4_ATTRS_TD
#define VC4_ATTRS_TD

include "vc4/Dialect/VC4/IR/VC4Base.td"
include "vc4/Dialect/VC4/IR/VC4Enums.td"

class VC4_DialectEnumAttr<EnumAttrInfo enumInfo, string name>
    : EnumAttr<VC4_Dialect, enumInfo, name> {
  let cppNamespace = "::mlir::vc4";
  let assemblyFormat = "`<` $value `>`";
}

def VC4_AddOpcodeAttr : VC4_DialectEnumAttr<VC4_AddOpcode, "add_opcode">;
def VC4_MulOpcodeAttr : VC4_DialectEnumAttr<VC4_MulOpcode, "mul_opcode">;
def VC4_CondAttr : VC4_DialectEnumAttr<VC4_Cond, "cond">;
def VC4_LoadImmModeAttr
    : VC4_DialectEnumAttr<VC4_LoadImmMode, "load_imm_mode">;
def VC4_RegfileAPackModeAttr
    : VC4_DialectEnumAttr<VC4_RegfileAPackMode, "regfile_a_pack_mode">;
def VC4_RegfileAUnpackModeAttr
    : VC4_DialectEnumAttr<VC4_RegfileAUnpackMode, "regfile_a_unpack_mode">;
def VC4_R4UnpackModeAttr
    : VC4_DialectEnumAttr<VC4_R4UnpackMode, "r4_unpack_mode">;
def VC4_MulPackModeAttr
    : VC4_DialectEnumAttr<VC4_MulPackMode, "mul_pack_mode">;

#endif // VC4_ATTRS_TD
