#ifndef VC4_ATTRS_TD
#define VC4_ATTRS_TD

include "vc4/Dialect/VC4/IR/VC4Base.td"
include "vc4/Dialect/VC4/IR/VC4Enums.td"
include "vc4/Dialect/VC4/IR/VC4OpEnums.td"

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
def VC4_QPUSignalAttr : VC4_DialectEnumAttr<VC4_QPUSignal, "qpu_signal">;
def VC4_QPUMuxAttr : VC4_DialectEnumAttr<VC4_QPUMux, "qpu_mux">;
def VC4_TMUUnitAttr : VC4_DialectEnumAttr<VC4_TMUUnit, "tmu_unit">;
def VC4_TMUModeAttr : VC4_DialectEnumAttr<VC4_TMUMode, "tmu_mode">;
def VC4_TextureTypeAttr
    : VC4_DialectEnumAttr<VC4_TextureType, "texture_type">;
def VC4_MagFilterAttr
    : VC4_DialectEnumAttr<VC4_MagFilter, "mag_filter">;
def VC4_MinFilterAttr
    : VC4_DialectEnumAttr<VC4_MinFilter, "min_filter">;
def VC4_WrapModeAttr : VC4_DialectEnumAttr<VC4_WrapMode, "wrap_mode">;
def VC4_TMUReadPartAttr
    : VC4_DialectEnumAttr<VC4_TMUReadPart, "tmu_read_part">;
def VC4_SFUKindAttr : VC4_DialectEnumAttr<VC4_SFUKind, "sfu_kind">;
def VC4_VPMDescKindAttr
    : VC4_DialectEnumAttr<VC4_VPMDescKind, "vpm_desc_kind">;
def VC4_VPMOrientationAttr
    : VC4_DialectEnumAttr<VC4_VPMOrientation, "vpm_orientation">;
def VC4_VPMLaneModeAttr
    : VC4_DialectEnumAttr<VC4_VPMLaneMode, "vpm_lane_mode">;
def VC4_VPMElemWidthAttr
    : VC4_DialectEnumAttr<VC4_VPMElemWidth, "vpm_elem_width">;
def VC4_DMADescKindAttr
    : VC4_DialectEnumAttr<VC4_DMADescKind, "dma_desc_kind">;
def VC4_DMABlockModeAttr
    : VC4_DialectEnumAttr<VC4_DMABlockMode, "dma_block_mode">;
def VC4_DMAOrientationAttr
    : VC4_DialectEnumAttr<VC4_DMAOrientation, "dma_orientation">;
def VC4_DMAElemWidthAttr
    : VC4_DialectEnumAttr<VC4_DMAElemWidth, "dma_elem_width">;
def VC4_SemaphoreModeAttr
    : VC4_DialectEnumAttr<VC4_SemaphoreMode, "semaphore_mode">;
def VC4_MutexModeAttr
    : VC4_DialectEnumAttr<VC4_MutexMode, "mutex_mode">;
def VC4_ThreadSwitchModeAttr
    : VC4_DialectEnumAttr<VC4_ThreadSwitchMode, "thread_switch_mode">;
def VC4_BranchCondAttr
    : VC4_DialectEnumAttr<VC4_BranchCond, "branch_cond">;
def VC4_V3DQueryKindAttr
    : VC4_DialectEnumAttr<VC4_V3DQueryKind, "v3d_query_kind">;
def VC4_V3DConfigureKindAttr
    : VC4_DialectEnumAttr<VC4_V3DConfigureKind, "v3d_configure_kind">;

#endif // VC4_ATTRS_TD
