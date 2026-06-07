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
      I32EnumAttrCase<"add", 0>,
      I32EnumAttrCase<"min_s", 1>,
      I32EnumAttrCase<"max_s", 2>,
      I32EnumAttrCase<"min_u", 3>,
      I32EnumAttrCase<"max_u", 4>,
      I32EnumAttrCase<"fmin", 5>,
      I32EnumAttrCase<"fmax", 6>,
      I32EnumAttrCase<"bit_and", 7>,
      I32EnumAttrCase<"bit_or", 8>,
      I32EnumAttrCase<"bit_xor", 9>
    ]>;

def VC4Kernel_CmpPredicate : VC4Kernel_DialectI32Enum<"CmpPredicate",
    "VC4Kernel fragment compare predicate", [
      I32EnumAttrCase<"eq", 0>,
      I32EnumAttrCase<"ne", 1>,
      I32EnumAttrCase<"ult", 2>,
      I32EnumAttrCase<"ule", 3>,
      I32EnumAttrCase<"ugt", 4>,
      I32EnumAttrCase<"uge", 5>,
      I32EnumAttrCase<"slt", 6>,
      I32EnumAttrCase<"sle", 7>,
      I32EnumAttrCase<"sgt", 8>,
      I32EnumAttrCase<"sge", 9>,
      I32EnumAttrCase<"oeq", 10>,
      I32EnumAttrCase<"one", 11>,
      I32EnumAttrCase<"olt", 12>,
      I32EnumAttrCase<"ole", 13>,
      I32EnumAttrCase<"ogt", 14>,
      I32EnumAttrCase<"oge", 15>,
      I32EnumAttrCase<"uno", 16>,
      I32EnumAttrCase<"ueq", 17>,
      I32EnumAttrCase<"une", 18>
    ]>;

def VC4Kernel_FPCmpPolicy : VC4Kernel_DialectI32Enum<"FPCmpPolicy",
    "VC4Kernel floating-point compare policy", [
      I32EnumAttrCase<"finite_only", 0>
    ]>;

def VC4Kernel_FPReducePolicy : VC4Kernel_DialectI32Enum<"FPReducePolicy",
    "VC4Kernel floating-point reduction policy", [
      I32EnumAttrCase<"finite_tree", 0>
    ]>;

def VC4Kernel_SFUKind : VC4Kernel_DialectI32Enum<"SFUKind",
    "VC4Kernel hardware special-functions unit kind", [
      I32EnumAttrCase<"recip", 52>,
      I32EnumAttrCase<"rsqrt", 53>,
      I32EnumAttrCase<"exp", 54>,
      I32EnumAttrCase<"log", 55>
    ]>;

def VC4Kernel_FPMathPolicy : VC4Kernel_DialectI32Enum<"FPMathPolicy",
    "VC4Kernel explicit approximate floating-point math policy", [
      I32EnumAttrCase<"approx_sfu", 0>,
      I32EnumAttrCase<"exact", 1>
    ]>;

def VC4Kernel_FPDomain : VC4Kernel_DialectI32Enum<"FPDomain",
    "VC4Kernel explicit finite floating-point domain contract", [
      I32EnumAttrCase<"finite", 0>,
      I32EnumAttrCase<"finite_nonzero", 1>,
      I32EnumAttrCase<"finite_positive", 2>
    ]>;

def VC4Kernel_AddALUOpcode : VC4Kernel_DialectI32Enum<"AddALUOpcode",
    "VC4Kernel ADD-pipe fragment ALU opcode", [
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

def VC4Kernel_MulALUOpcode : VC4Kernel_DialectI32Enum<"MulALUOpcode",
    "VC4Kernel MUL-pipe fragment ALU opcode", [
      I32EnumAttrCase<"nop", 0>,
      I32EnumAttrCase<"fmul", 1>,
      I32EnumAttrCase<"mul24", 2>,
      I32EnumAttrCase<"v8muld", 3>,
      I32EnumAttrCase<"v8min", 4>,
      I32EnumAttrCase<"v8max", 5>,
      I32EnumAttrCase<"v8adds", 6>,
      I32EnumAttrCase<"v8subs", 7>
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

def VC4Kernel_SubwordType : VC4Kernel_DialectI32Enum<"SubwordType",
    "VC4Kernel explicit fragment pack/unpack subword type", [
      I32EnumAttrCase<"u8", 0>,
      I32EnumAttrCase<"s8", 1>,
      I32EnumAttrCase<"u16", 2>,
      I32EnumAttrCase<"s16", 3>
    ]>;

def VC4Kernel_SubwordLayout : VC4Kernel_DialectI32Enum<"SubwordLayout",
    "VC4Kernel explicit fragment pack/unpack subword layout", [
      I32EnumAttrCase<"packed", 0>,
      I32EnumAttrCase<"laned", 1>
    ]>;

def VC4Kernel_UnpackPolicy : VC4Kernel_DialectI32Enum<"UnpackPolicy",
    "VC4Kernel explicit fragment unpack policy", [
      I32EnumAttrCase<"zero_extend", 0>,
      I32EnumAttrCase<"sign_extend", 1>
    ]>;

def VC4Kernel_PackPolicy : VC4Kernel_DialectI32Enum<"PackPolicy",
    "VC4Kernel explicit fragment pack policy", [
      I32EnumAttrCase<"truncate", 0>,
      I32EnumAttrCase<"saturate", 1>,
      I32EnumAttrCase<"wrap", 2>
    ]>;

def VC4Kernel_MemoryPath : VC4Kernel_DialectI32Enum<"MemoryPath",
    "VC4Kernel explicit memory path", [
      I32EnumAttrCase<"vpm_qpu", 0>,
      I32EnumAttrCase<"tmu_global_read", 1>,
      I32EnumAttrCase<"vdr_global_to_vpm", 2>,
      I32EnumAttrCase<"vdw_global_store", 3>,
      I32EnumAttrCase<"compiler_spill_vdw_vdr", 4>
    ]>;

def VC4Kernel_Coherency : VC4Kernel_DialectI32Enum<"Coherency",
    "VC4Kernel memory coherency domain", [
      I32EnumAttrCase<"vpm_local", 0>,
      I32EnumAttrCase<"readonly_tmu", 1>,
      I32EnumAttrCase<"dma_ordered", 2>,
      I32EnumAttrCase<"compiler_spill_coherent", 3>
    ]>;

def VC4Kernel_InactiveLoad : VC4Kernel_DialectI32Enum<"InactiveLoad",
    "VC4Kernel inactive-load policy", [
      I32EnumAttrCase<"zero", 0>
    ]>;

def VC4Kernel_InactiveStore : VC4Kernel_DialectI32Enum<"InactiveStore",
    "VC4Kernel inactive-store policy", [
      I32EnumAttrCase<"preserve", 0>,
      I32EnumAttrCase<"deterministic_reject_sparse", 1>
    ]>;

def VC4Kernel_ScheduleModeAttr
    : VC4Kernel_EnumAttr<VC4Kernel_ScheduleMode, "schedule_mode">;
def VC4Kernel_ReduceKindAttr
    : VC4Kernel_EnumAttr<VC4Kernel_ReduceKind, "reduce">;
def VC4Kernel_CmpPredicateAttr
    : VC4Kernel_EnumAttr<VC4Kernel_CmpPredicate, "cmp">;
def VC4Kernel_FPCmpPolicyAttr
    : VC4Kernel_EnumAttr<VC4Kernel_FPCmpPolicy, "fp_cmp_policy">;
def VC4Kernel_FPReducePolicyAttr
    : VC4Kernel_EnumAttr<VC4Kernel_FPReducePolicy, "fp_reduce_policy">;
def VC4Kernel_SFUKindAttr
    : VC4Kernel_EnumAttr<VC4Kernel_SFUKind, "sfu_kind">;
def VC4Kernel_FPMathPolicyAttr
    : VC4Kernel_EnumAttr<VC4Kernel_FPMathPolicy, "fp_math_policy">;
def VC4Kernel_FPDomainAttr
    : VC4Kernel_EnumAttr<VC4Kernel_FPDomain, "fp_domain">;
def VC4Kernel_AddALUOpcodeAttr
    : VC4Kernel_EnumAttr<VC4Kernel_AddALUOpcode, "add_alu_opcode">;
def VC4Kernel_MulALUOpcodeAttr
    : VC4Kernel_EnumAttr<VC4Kernel_MulALUOpcode, "mul_alu_opcode">;
def VC4Kernel_VPMOrientationAttr
    : VC4Kernel_EnumAttr<VC4Kernel_VPMOrientation, "vpm_orientation">;
def VC4Kernel_VPMWidthAttr
    : VC4Kernel_EnumAttr<VC4Kernel_VPMWidth, "vpm_width">;
def VC4Kernel_VPMSubwordAttr
    : VC4Kernel_EnumAttr<VC4Kernel_VPMSubword, "vpm_subword">;
def VC4Kernel_SubwordTypeAttr
    : VC4Kernel_EnumAttr<VC4Kernel_SubwordType, "subword_type">;
def VC4Kernel_SubwordLayoutAttr
    : VC4Kernel_EnumAttr<VC4Kernel_SubwordLayout, "subword_layout">;
def VC4Kernel_UnpackPolicyAttr
    : VC4Kernel_EnumAttr<VC4Kernel_UnpackPolicy, "unpack_policy">;
def VC4Kernel_PackPolicyAttr
    : VC4Kernel_EnumAttr<VC4Kernel_PackPolicy, "pack_policy">;
def VC4Kernel_MemoryPathAttr
    : VC4Kernel_EnumAttr<VC4Kernel_MemoryPath, "memory_path">;
def VC4Kernel_CoherencyAttr
    : VC4Kernel_EnumAttr<VC4Kernel_Coherency, "coherency">;
def VC4Kernel_InactiveLoadAttr
    : VC4Kernel_EnumAttr<VC4Kernel_InactiveLoad, "inactive_load">;
def VC4Kernel_InactiveStoreAttr
    : VC4Kernel_EnumAttr<VC4Kernel_InactiveStore, "inactive_store">;

#endif // VC4KERNEL_ATTRS_TD
