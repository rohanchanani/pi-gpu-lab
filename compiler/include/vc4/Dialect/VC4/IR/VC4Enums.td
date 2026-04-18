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

def VC4_TMUUnit : VC4_DialectI32Enum<"TMUUnit",
    "VC4 TMU unit selection", [
      I32EnumAttrCase<"tmu0", 0>,
      I32EnumAttrCase<"tmu1", 1>
    ]>;

def VC4_TMUMode : VC4_DialectI32Enum<"TMUMode",
    "VC4 TMU descriptor mode", [
      I32EnumAttrCase<"direct", 0>,
      I32EnumAttrCase<"texture2d", 1>,
      I32EnumAttrCase<"cubemap", 2>
    ]>;

def VC4_TextureType : VC4_DialectI32Enum<"TextureType",
    "VC4 texture format selection", [
      I32EnumAttrCase<"rgba8888", 0>,
      I32EnumAttrCase<"rgbx8888", 1>,
      I32EnumAttrCase<"rgba4444", 2>,
      I32EnumAttrCase<"rgba5551", 3>,
      I32EnumAttrCase<"rgb565", 4>,
      I32EnumAttrCase<"luminance", 5>,
      I32EnumAttrCase<"alpha", 6>,
      I32EnumAttrCase<"lumalpha", 7>,
      I32EnumAttrCase<"etc1", 8>,
      I32EnumAttrCase<"s16f", 9>,
      I32EnumAttrCase<"s8", 10>,
      I32EnumAttrCase<"s16", 11>,
      I32EnumAttrCase<"bw1", 12>,
      I32EnumAttrCase<"a4", 13>,
      I32EnumAttrCase<"a1", 14>,
      I32EnumAttrCase<"rgba64", 15>,
      I32EnumAttrCase<"rgba32r", 16>,
      I32EnumAttrCase<"yuyv422r", 17>
    ]>;

def VC4_MagFilter : VC4_DialectI32Enum<"MagFilter",
    "VC4 TMU magnification filter", [
      I32EnumAttrCase<"linear", 0>,
      I32EnumAttrCase<"nearest", 1>
    ]>;

def VC4_MinFilter : VC4_DialectI32Enum<"MinFilter",
    "VC4 TMU minification and mip filter", [
      I32EnumAttrCase<"linear", 0>,
      I32EnumAttrCase<"nearest", 1>,
      I32EnumAttrCase<"near_mip_near", 2>,
      I32EnumAttrCase<"near_mip_lin", 3>,
      I32EnumAttrCase<"lin_mip_near", 4>,
      I32EnumAttrCase<"lin_mip_lin", 5>
    ]>;

def VC4_WrapMode : VC4_DialectI32Enum<"WrapMode",
    "VC4 TMU wrap mode", [
      I32EnumAttrCase<"repeat", 0>,
      I32EnumAttrCase<"clamp", 1>,
      I32EnumAttrCase<"mirror", 2>,
      I32EnumAttrCase<"border", 3>
    ]>;

def VC4_TMUReadPart : VC4_DialectI32Enum<"TMUReadPart",
    "VC4 TMU receive selection", [
      I32EnumAttrCase<"raw32", 0>,
      I32EnumAttrCase<"rgba8888", 1>,
      I32EnumAttrCase<"rg1616", 2>,
      I32EnumAttrCase<"ba1616", 3>
    ]>;

def VC4_SFUKind : VC4_DialectI32Enum<"SFUKind",
    "VC4 SFU operation kind", [
      I32EnumAttrCase<"recip", 0>,
      I32EnumAttrCase<"recipsqrt", 1>,
      I32EnumAttrCase<"exp", 2>,
      I32EnumAttrCase<"log", 3>
    ]>;

def VC4_VPMDescKind : VC4_DialectI32Enum<"VPMDescKind",
    "VC4 VPM descriptor direction", [
      I32EnumAttrCase<"read", 0>,
      I32EnumAttrCase<"write", 1>
    ]>;

def VC4_VPMOrientation : VC4_DialectI32Enum<"VPMOrientation",
    "VC4 VPM block orientation", [
      I32EnumAttrCase<"horizontal", 0>,
      I32EnumAttrCase<"vertical", 1>
    ]>;

def VC4_VPMLaneMode : VC4_DialectI32Enum<"VPMLaneMode",
    "VC4 VPM lane layout", [
      I32EnumAttrCase<"packed", 0>,
      I32EnumAttrCase<"laned", 1>
    ]>;

def VC4_VPMElemWidth : VC4_DialectI32Enum<"VPMElemWidth",
    "VC4 VPM element width", [
      I32EnumAttrCase<"w8", 0>,
      I32EnumAttrCase<"w16", 1>,
      I32EnumAttrCase<"w32", 2>
    ]>;

def VC4_DMADescKind : VC4_DialectI32Enum<"DMADescKind",
    "VC4 DMA descriptor direction", [
      I32EnumAttrCase<"load", 0>,
      I32EnumAttrCase<"store", 1>
    ]>;

def VC4_DMABlockMode : VC4_DialectI32Enum<"DMABlockMode",
    "VC4 DMA block layout mode", [
      I32EnumAttrCase<"row_row", 0>,
      I32EnumAttrCase<"packed_rows", 1>
    ]>;

def VC4_DMAOrientation : VC4_DialectI32Enum<"DMAOrientation",
    "VC4 DMA orientation", [
      I32EnumAttrCase<"horizontal", 0>,
      I32EnumAttrCase<"vertical", 1>
    ]>;

def VC4_DMAElemWidth : VC4_DialectI32Enum<"DMAElemWidth",
    "VC4 DMA transfer element width", [
      I32EnumAttrCase<"w8", 0>,
      I32EnumAttrCase<"w16", 1>,
      I32EnumAttrCase<"w32", 2>
    ]>;

def VC4_SemaphoreMode : VC4_DialectI32Enum<"SemaphoreMode",
    "VC4 hardware semaphore mode", [
      I32EnumAttrCase<"acquire", 0>,
      I32EnumAttrCase<"release", 1>
    ]>;

def VC4_MutexMode : VC4_DialectI32Enum<"MutexMode",
    "VC4 hardware mutex mode", [
      I32EnumAttrCase<"acquire", 0>,
      I32EnumAttrCase<"release", 1>
    ]>;

def VC4_ThreadSwitchMode : VC4_DialectI32Enum<"ThreadSwitchMode",
    "VC4 hardware thread-switch mode", [
      I32EnumAttrCase<"thread_switch", 0, "switch">,
      I32EnumAttrCase<"last_switch", 1>
    ]>;

#endif // VC4_ENUMS_TD
