#ifndef VC4_ENUMS_TD
#define VC4_ENUMS_TD

include "mlir/IR/EnumAttr.td"
include "vc4/Dialect/VC4/IR/VC4Base.td"

class VC4_I32Enum<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::vc4";
}

class VC4_DialectI32Enum<string name, string summary,
                         list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::vc4";
  let genSpecializedAttr = 0;
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

def VC4_ThreadingMode : VC4_I32Enum<"ThreadingMode",
    "VC4 function threading mode", [
      I32EnumAttrCase<"single", 0>,
      I32EnumAttrCase<"threadable", 1>
    ]>;

def VC4_BuiltinKind : VC4_I32Enum<"BuiltinKind",
    "VC4 builtin kind", [
      I32EnumAttrCase<"elem_num", 0>,
      I32EnumAttrCase<"qpu_num", 1>
    ]>;

// Encodings match the VideoCore IV 3D Architecture Reference Guide QPU
// instruction tables for encoded ALU, condition, immediate, and pack/unpack
// fields.

def VC4_AddOpcode : VC4_DialectI32Enum<"AddOpcode",
    "VC4 add ALU opcode", [
      I32EnumAttrCase<"nop", 0>,
      I32EnumAttrCase<"fadd", 1>,
      I32EnumAttrCase<"fsub", 2>,
      I32EnumAttrCase<"fmin", 3>,
      I32EnumAttrCase<"fmax", 4>,
      I32EnumAttrCase<"fminabs", 5>,
      I32EnumAttrCase<"fmaxabs", 6>,
      I32EnumAttrCase<"ftoi", 7>,
      I32EnumAttrCase<"itof", 8>,
      I32EnumAttrCase<"add", 12>,
      I32EnumAttrCase<"sub", 13>,
      I32EnumAttrCase<"shr", 14>,
      I32EnumAttrCase<"asr", 15>,
      I32EnumAttrCase<"ror", 16>,
      I32EnumAttrCase<"shl", 17>,
      I32EnumAttrCase<"min", 18>,
      I32EnumAttrCase<"max", 19>,
      I32EnumAttrCase<"bit_and", 20, "and">,
      I32EnumAttrCase<"bit_or", 21, "or">,
      I32EnumAttrCase<"bit_xor", 22, "xor">,
      I32EnumAttrCase<"bit_not", 23, "not">,
      I32EnumAttrCase<"clz", 24>,
      I32EnumAttrCase<"v8adds", 30>,
      I32EnumAttrCase<"v8subs", 31>
    ]>;

def VC4_MulOpcode : VC4_DialectI32Enum<"MulOpcode",
    "VC4 mul ALU opcode", [
      I32EnumAttrCase<"nop", 0>,
      I32EnumAttrCase<"fmul", 1>,
      I32EnumAttrCase<"mul24", 2>,
      I32EnumAttrCase<"v8muld", 3>,
      I32EnumAttrCase<"v8min", 4>,
      I32EnumAttrCase<"v8max", 5>,
      I32EnumAttrCase<"v8adds", 6>,
      I32EnumAttrCase<"v8subs", 7>
    ]>;

def VC4_Cond : VC4_DialectI32Enum<"Cond",
    "VC4 ALU condition code", [
      I32EnumAttrCase<"never", 0>,
      I32EnumAttrCase<"always", 1>,
      I32EnumAttrCase<"zs", 2>,
      I32EnumAttrCase<"zc", 3>,
      I32EnumAttrCase<"ns", 4>,
      I32EnumAttrCase<"nc", 5>,
      I32EnumAttrCase<"cs", 6>,
      I32EnumAttrCase<"cc", 7>
    ]>;

def VC4_LoadImmMode : VC4_DialectI32Enum<"LoadImmMode",
    "VC4 load-immediate encoding mode", [
      I32EnumAttrCase<"splat32", 0>,
      I32EnumAttrCase<"per_elem_i2", 1>,
      I32EnumAttrCase<"per_elem_u2", 3>
    ]>;

def VC4_RegfileAUnpackMode : VC4_DialectI32Enum<"RegfileAUnpackMode",
    "VC4 regfile-A unpack mode", [
      I32EnumAttrCase<"none", 0>,
      I32EnumAttrCase<"f16a_or_i16a", 1>,
      I32EnumAttrCase<"f16b_or_i16b", 2>,
      I32EnumAttrCase<"replicate_8d", 3>,
      I32EnumAttrCase<"color8a", 4>,
      I32EnumAttrCase<"color8b", 5>,
      I32EnumAttrCase<"color8c", 6>,
      I32EnumAttrCase<"color8d", 7>
    ]>;

def VC4_RegfileAPackMode : VC4_DialectI32Enum<"RegfileAPackMode",
    "VC4 regfile-A pack mode", [
      I32EnumAttrCase<"none", 0>,
      I32EnumAttrCase<"to_16a", 1>,
      I32EnumAttrCase<"to_16b", 2>,
      I32EnumAttrCase<"to_8888", 3>,
      I32EnumAttrCase<"to_8a", 4>,
      I32EnumAttrCase<"to_8b", 5>,
      I32EnumAttrCase<"to_8c", 6>,
      I32EnumAttrCase<"to_8d", 7>,
      I32EnumAttrCase<"sat32", 8>,
      I32EnumAttrCase<"sat16a", 9>,
      I32EnumAttrCase<"sat16b", 10>,
      I32EnumAttrCase<"sat8888", 11>,
      I32EnumAttrCase<"sat8a", 12>,
      I32EnumAttrCase<"sat8b", 13>,
      I32EnumAttrCase<"sat8c", 14>,
      I32EnumAttrCase<"sat8d", 15>
    ]>;

def VC4_R4UnpackMode : VC4_DialectI32Enum<"R4UnpackMode",
    "VC4 r4 unpack mode", [
      I32EnumAttrCase<"none", 0>,
      I32EnumAttrCase<"f16a", 1>,
      I32EnumAttrCase<"f16b", 2>,
      I32EnumAttrCase<"replicate_8d", 3>,
      I32EnumAttrCase<"color8a", 4>,
      I32EnumAttrCase<"color8b", 5>,
      I32EnumAttrCase<"color8c", 6>,
      I32EnumAttrCase<"color8d", 7>
    ]>;

def VC4_MulPackMode : VC4_DialectI32Enum<"MulPackMode",
    "VC4 mul ALU pack mode", [
      I32EnumAttrCase<"none", 0>,
      I32EnumAttrCase<"to_8888", 3>,
      I32EnumAttrCase<"to_8a", 4>,
      I32EnumAttrCase<"to_8b", 5>,
      I32EnumAttrCase<"to_8c", 6>,
      I32EnumAttrCase<"to_8d", 7>
    ]>;

#endif // VC4_ENUMS_TD
