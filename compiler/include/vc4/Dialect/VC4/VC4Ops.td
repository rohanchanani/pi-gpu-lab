include "vc4/Dialect/VC4/VC4Dialect.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VC4_Op<string mnemonic, list<Trait> traits = []> :
    Op<VC4_Dialect, mnemonic, traits>;

def VC4_UniformResource : Resource<"::mlir::vc4::UniformResource">;
def VC4_VPMResource : Resource<"::mlir::vc4::VPMResource">;
def VC4_GlobalMemoryResource : Resource<"::mlir::vc4::GlobalMemoryResource">;
def VC4_Vec16F32 : Type<
    CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
          "::llvm::cast<::mlir::VectorType>($_self).getShape() == "
          "::llvm::ArrayRef<int64_t>{16} && "
          "::llvm::cast<::mlir::VectorType>($_self).getElementType().isF32()">,
    "vector<16xf32>">;

def VC4_GetBuiltinOp : VC4_Op<"get_builtin", [Pure]> {
  let summary = "Read a physical VC4 execution builtin";
  let description = [{
    Reads a physical execution builtin such as the current QPU id or the launch
    width in QPUs. These are target-side execution facts, not user kernel
    arguments.
  }];
  let arguments = (ins StrAttr:$kind);
  let results = (outs Index:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::vc4::BuiltinKind getBuiltinKind();
  }];
}

def VC4_GetUniformOp : VC4_Op<"get_uniform", [
    MemoryEffects<[MemRead<VC4_UniformResource>]>]> {
  let summary = "Read a kernel uniform by slot";
  let description = [{
    Reads one uniform / kernel parameter and returns it as a typed SSA value.
    The slot number is a compact stand-in for the eventual launch convention.
  }];
  let arguments = (ins I32Attr:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`[` $index `]` attr-dict `:` type($result)";
}

def VC4_DmaLoadOp : VC4_Op<"dma_load", [
    MemoryEffects<[MemRead<VC4_GlobalMemoryResource>,
                   MemWrite<VC4_VPMResource>]>]> {
  let summary = "Stage data from global memory into VPM";
  let description = [{
    Represents a global-to-VPM DMA transfer from a scalar-element memref view at
    a given global element offset into a selected VPM row.
  }];
  let arguments = (ins AnyMemRef:$source, Index:$globalOffset, Index:$row);
  let assemblyFormat = [{
    $source `[` $globalOffset `]` `to` $row attr-dict `:`
    type($source) `,` type($globalOffset) `,` type($row)
  }];
}

def VC4_DmaStoreOp : VC4_Op<"dma_store", [
    MemoryEffects<[MemRead<VC4_VPMResource>,
                   MemWrite<VC4_GlobalMemoryResource>]>]> {
  let summary = "Store staged data from VPM back to global memory";
  let description = [{
    Represents a VPM-to-global DMA transfer from a selected VPM row into a
    scalar-element memref view at a given global element offset.
  }];
  let arguments = (ins AnyMemRef:$target, Index:$globalOffset, Index:$row);
  let assemblyFormat = [{
    $row `to` $target `[` $globalOffset `]` attr-dict `:`
    type($row) `,` type($target) `,` type($globalOffset)
  }];
}

def VC4_VPMReadOp : VC4_Op<"vpm_read", [
    MemoryEffects<[MemRead<VC4_VPMResource>]>]> {
  let summary = "Read a vector value from a VPM row";
  let description = [{
    Reads a staged vector from a VPM row into SSA form so upstream vector and
    arithmetic dialects can operate on it.
  }];
  let arguments = (ins Index:$row);
  let results = (outs AnyVectorOfAnyRank:$result);
  let assemblyFormat = "`from` $row attr-dict `:` type($row) `->` type($result)";
}

def VC4_VPMWriteOp : VC4_Op<"vpm_write", [
    MemoryEffects<[MemWrite<VC4_VPMResource>]>]> {
  let summary = "Write a vector value into a VPM row";
  let description = [{
    Writes a vector SSA value into a VPM row, making the staged result available
    for later DMA store or additional staged-memory operations.
  }];
  let arguments = (ins Index:$row, AnyVectorOfAnyRank:$value);
  let assemblyFormat =
      "$value `to` $row attr-dict `:` type($value) `,` type($row)";
}

def VC4_FMulOp : VC4_Op<"fmul", [Pure]> {
  let summary = "Multiply two staged VC4 vectors";
  let description = [{
    Represents the target-side floating-point vector multiply needed for the
    first SAXPY lowering path.
  }];
  let arguments = (ins VC4_Vec16F32:$lhs, VC4_Vec16F32:$rhs);
  let results = (outs VC4_Vec16F32:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VC4_FAddOp : VC4_Op<"fadd", [Pure]> {
  let summary = "Add two staged VC4 vectors";
  let description = [{
    Represents the target-side floating-point vector add needed for the first
    SAXPY lowering path.
  }];
  let arguments = (ins VC4_Vec16F32:$lhs, VC4_Vec16F32:$rhs);
  let results = (outs VC4_Vec16F32:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}
