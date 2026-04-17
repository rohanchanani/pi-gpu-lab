include "vc4/Dialect/VC4/VC4Dialect.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VC4_Op<string mnemonic, list<Trait> traits = []> :
    Op<VC4_Dialect, mnemonic, traits>;

def VC4_UniformResource : Resource<"::mlir::vc4::UniformResource">;
def VC4_StagingResource : Resource<"::mlir::vc4::StagingResource">;
def VC4_GlobalMemoryResource : Resource<"::mlir::vc4::GlobalMemoryResource">;
def VC4_Vec16F32 : Type<
    CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
          "::llvm::cast<::mlir::VectorType>($_self).getShape() == "
          "::llvm::ArrayRef<int64_t>{16} && "
          "::llvm::cast<::mlir::VectorType>($_self).getElementType().isF32()">,
    "vector<16xf32>">;

def VC4_GetBuiltinOp : VC4_Op<"get_builtin", [Pure]> {
  let summary = "Read a backend-introduced execution builtin";
  let description = [{
    Reads a physical execution builtin such as the current QPU id or the launch
    width in QPUs. These are target-side execution facts introduced by backend
    lowering, not user kernel arguments.
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
  let summary = "Read a user/kernel uniform by logical ABI position";
  let description = [{
    Reads one logical user/kernel argument from the uniform stream and returns
    it as a typed SSA value. The index is a logical ABI position in source
    signature order; it is not a promise about physical random-access storage.
  }];
  let arguments = (ins I32Attr:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`[` $index `]` attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4_DmaLoadOp : VC4_Op<"dma_load", [
    MemoryEffects<[MemRead<VC4_GlobalMemoryResource>,
                   MemWrite<VC4_StagingResource>]>]> {
  let summary = "Stage data from global memory into an abstract staging slot";
  let description = [{
    Represents a global-to-staging transfer from a scalar-element memref view at
    a given global element offset into an abstract staging slot.

    The slot identifies a compiler-visible virtual staging location. It does
    not encode a physical VPM row number. Final vc4asm emission is responsible
    for mapping slots onto physical rows.
  }];
  let arguments = (ins AnyMemRef:$source, Index:$globalOffset, I32Attr:$slot);
  let assemblyFormat = [{
    $source `[` $globalOffset `]` `to` `slot` `[` $slot `]` attr-dict `:`
    type($source) `,` type($globalOffset)
  }];
  let hasVerifier = 1;
}

def VC4_DmaStoreOp : VC4_Op<"dma_store", [
    MemoryEffects<[MemRead<VC4_StagingResource>,
                   MemWrite<VC4_GlobalMemoryResource>]>]> {
  let summary = "Store staged data from an abstract staging slot to global memory";
  let description = [{
    Represents a staging-to-global transfer from an abstract staging slot into a
    scalar-element memref view at a given global element offset.

    As with vc4.dma_load, the slot is a virtual staging location rather than a
    physical VPM row number.
  }];
  let arguments = (ins I32Attr:$slot, AnyMemRef:$target, Index:$globalOffset);
  let assemblyFormat = [{
    `slot` `[` $slot `]` `to` $target `[` $globalOffset `]` attr-dict `:`
    type($target) `,` type($globalOffset)
  }];
  let hasVerifier = 1;
}

def VC4_StagedReadOp : VC4_Op<"staged_read", [
    MemoryEffects<[MemRead<VC4_StagingResource>]>]> {
  let summary = "Read a staged vector value from an abstract staging slot";
  let description = [{
    Reads a vector value from an abstract staging slot into SSA form so generic
    arithmetic and dataflow ops can consume it.

    The slot is virtual and compiler-facing. It is not a physical VPM row.
  }];
  let arguments = (ins I32Attr:$slot);
  let results = (outs AnyVectorOfAnyRank:$result);
  let assemblyFormat =
      "`from` `slot` `[` $slot `]` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def VC4_StagedWriteOp : VC4_Op<"staged_write", [
    MemoryEffects<[MemWrite<VC4_StagingResource>]>]> {
  let summary = "Write a vector value into an abstract staging slot";
  let description = [{
    Writes a vector SSA value into an abstract staging slot, making the staged
    result available for later staged-memory operations.

    The slot is virtual and compiler-facing. It is not a physical VPM row.
  }];
  let arguments = (ins I32Attr:$slot, AnyVectorOfAnyRank:$value);
  let assemblyFormat =
      "`slot` `[` $slot `]` `=` $value attr-dict `:` type($value)";
  let hasVerifier = 1;
}

def VC4_FMulOp : VC4_Op<"fmul", [Pure]> {
  let summary = "Multiply two currently supported VC4 vector values";
  let description = [{
    Represents target-side floating-point vector multiply for the currently
    supported straight-line compute subset.
  }];
  let arguments = (ins VC4_Vec16F32:$lhs, VC4_Vec16F32:$rhs);
  let results = (outs VC4_Vec16F32:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VC4_FAddOp : VC4_Op<"fadd", [Pure]> {
  let summary = "Add two currently supported VC4 vector values";
  let description = [{
    Represents target-side floating-point vector add for the currently supported
    straight-line compute subset.
  }];
  let arguments = (ins VC4_Vec16F32:$lhs, VC4_Vec16F32:$rhs);
  let results = (outs VC4_Vec16F32:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}
