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
  let assemblyFormat = "`<` $value `>`";
}

def VC4_FunctionForm : VC4_DialectI32Enum<"FunctionForm",
    "VC4 function form", [
      I32EnumAttrCase<"scheduled", 1>
    ]>;

def VC4_ThreadingMode : VC4_DialectI32Enum<"ThreadingMode",
    "VC4 function threading mode", [
      I32EnumAttrCase<"single", 0>,
      I32EnumAttrCase<"threadable", 1>
    ]>;

def VC4_ExecutionDomain : VC4_DialectI32Enum<"ExecutionDomain",
    "VC4 function execution domain", [
      I32EnumAttrCase<"qpu", 0>,
      I32EnumAttrCase<"host", 1>
    ]>;

def VC4_BuiltinKind : VC4_DialectI32Enum<"BuiltinKind",
    "VC4 builtin kind", [
      I32EnumAttrCase<"logical_request", 0>,
      I32EnumAttrCase<"total_requests", 1>,
      I32EnumAttrCase<"logical_block_id", 2>,
      I32EnumAttrCase<"logical_warp_id", 3>,
      I32EnumAttrCase<"warps_per_block", 4>,
      I32EnumAttrCase<"vpm_base_row", 5>,
      I32EnumAttrCase<"vpm_rows", 6>,
      I32EnumAttrCase<"semaphore_base", 7>,
      I32EnumAttrCase<"barrier_arrive_sem", 8>,
      I32EnumAttrCase<"barrier_go_sem", 9>,
      I32EnumAttrCase<"barrier_depart_sem", 10>,
      I32EnumAttrCase<"barrier_reset_sem", 11>,
      I32EnumAttrCase<"resident_request_id", 12>,
      I32EnumAttrCase<"spill_frame_base", 13>,
      I32EnumAttrCase<"spill_frame_bytes", 14>,
      I32EnumAttrCase<"spill_frame_stride_bytes", 15>,
      I32EnumAttrCase<"spill_vpm_row", 16>
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
      // Use the full hardware instruction-form encodings, not dense mode IDs.
      I32EnumAttrCase<"splat32", 112>,
      I32EnumAttrCase<"per_elem_i2", 113>,
      I32EnumAttrCase<"per_elem_u2", 115>
    ]>;

// These unpack enums keep the real field encodings. Splitting regfile-A and r4
// unpack into separate dialect enums does not imply renumbering the retained
// hardware values.
def VC4_RegfileAUnpackMode : VC4_DialectI32Enum<"RegfileAUnpackMode",
    "VC4 regfile-A unpack mode", [
      I32EnumAttrCase<"none", 0>,
      I32EnumAttrCase<"f16a_or_i16a", 2>,
      I32EnumAttrCase<"f16b_or_i16b", 4>,
      I32EnumAttrCase<"replicate_8d", 6>,
      I32EnumAttrCase<"color8a", 8>,
      I32EnumAttrCase<"color8b", 10>,
      I32EnumAttrCase<"color8c", 12>,
      I32EnumAttrCase<"color8d", 14>
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
      I32EnumAttrCase<"none", 1>,
      I32EnumAttrCase<"f16a", 3>,
      I32EnumAttrCase<"f16b", 5>,
      I32EnumAttrCase<"replicate_8d", 7>,
      I32EnumAttrCase<"color8a", 9>,
      I32EnumAttrCase<"color8b", 11>,
      I32EnumAttrCase<"color8c", 13>,
      I32EnumAttrCase<"color8d", 15>
    ]>;

// Pack values likewise preserve the real encoded field values. The MUL pack
// cases retain the PM-selected encodings instead of being renumbered to 0..N.
def VC4_MulPackMode : VC4_DialectI32Enum<"MulPackMode",
    "VC4 mul ALU pack mode", [
      I32EnumAttrCase<"none", 16>,
      I32EnumAttrCase<"to_8888", 19>,
      I32EnumAttrCase<"to_8a", 20>,
      I32EnumAttrCase<"to_8b", 21>,
      I32EnumAttrCase<"to_8c", 22>,
      I32EnumAttrCase<"to_8d", 23>
    ]>;

// Compute-focused scope omits fragment/tile-buffer-only signals, but the
// remaining signal values still use the original hardware numbering.
def VC4_QPUSignal : VC4_DialectI32Enum<"QPUSignal",
    "VC4 scheduled QPU signal field", [
      I32EnumAttrCase<"bkpt", 0>,
      I32EnumAttrCase<"none", 1>,
      I32EnumAttrCase<"thrsw", 2>,
      I32EnumAttrCase<"thrend", 3>,
      I32EnumAttrCase<"last_thread_switch", 6>,
      I32EnumAttrCase<"ldtmu0", 10>,
      I32EnumAttrCase<"ldtmu1", 11>,
      I32EnumAttrCase<"small_imm", 13>,
      I32EnumAttrCase<"load_imm", 14>,
      I32EnumAttrCase<"branch", 15>
    ]>;

def VC4_QPUMux : VC4_DialectI32Enum<"QPUMux",
    "VC4 scheduled QPU source mux", [
      I32EnumAttrCase<"r0", 0>,
      I32EnumAttrCase<"r1", 1>,
      I32EnumAttrCase<"r2", 2>,
      I32EnumAttrCase<"r3", 3>,
      I32EnumAttrCase<"r4", 4>,
      I32EnumAttrCase<"r5", 5>,
      I32EnumAttrCase<"a", 6>,
      I32EnumAttrCase<"b", 7>
    ]>;

def VC4_SemaphoreMode : VC4_DialectI32Enum<"SemaphoreMode",
    "VC4 hardware semaphore mode", [
      I32EnumAttrCase<"acquire", 0>,
      I32EnumAttrCase<"release", 1>
    ]>;

def VC4_BranchCond : VC4_DialectI32Enum<"BranchCond",
    "VC4 QPU branch condition", [
      I32EnumAttrCase<"all_z_set", 0>,
      I32EnumAttrCase<"all_z_clear", 1>,
      I32EnumAttrCase<"any_z_set", 2>,
      I32EnumAttrCase<"any_z_clear", 3>,
      I32EnumAttrCase<"all_n_set", 4>,
      I32EnumAttrCase<"all_n_clear", 5>,
      I32EnumAttrCase<"any_n_set", 6>,
      I32EnumAttrCase<"any_n_clear", 7>,
      I32EnumAttrCase<"all_c_set", 8>,
      I32EnumAttrCase<"all_c_clear", 9>,
      I32EnumAttrCase<"any_c_set", 10>,
      I32EnumAttrCase<"any_c_clear", 11>,
      I32EnumAttrCase<"always", 15>
    ]>;

def VC4_VPMVCDSide : VC4_DialectI32Enum<"VPMVCDSide",
    "VC4 VPM/VCD/VDW transfer side", [
      I32EnumAttrCase<"read", 0>,
      I32EnumAttrCase<"write", 1>
    ]>;

#endif // VC4_ENUMS_TD
