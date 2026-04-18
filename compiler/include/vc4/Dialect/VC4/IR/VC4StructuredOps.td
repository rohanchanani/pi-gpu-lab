#ifndef VC4_STRUCTURED_OPS_TD
#define VC4_STRUCTURED_OPS_TD

include "vc4/Dialect/VC4/IR/VC4Base.td"
include "vc4/Dialect/VC4/IR/VC4Attrs.td"
include "vc4/Dialect/VC4/IR/VC4OpEnums.td"
include "vc4/Dialect/VC4/IR/VC4Types.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VC4_Op<string mnemonic, list<Trait> traits = []>
    : Op<VC4_Dialect, mnemonic, traits>;

def VC4_ModuleOp : VC4_Op<"module", [
    IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol
  ] # GraphRegionNoTerminator.traits> {
  let summary = "VC4 target module";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility
  );
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";
  let extraClassDeclaration = [{
    static ModuleOp create(Location loc, StringRef name);
  }];
  let hasVerifier = 1;
}

def VC4_FuncOp : VC4_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
  ]> {
  let summary = "VC4 target function";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<UnitAttr>:$kernel,
    OptionalAttr<VC4_ThreadingModeAttr>:$threading,
    OptionalAttr<VC4_FunctionFormAttr>:$form
  );
  let regions = (region AnyRegion:$body);
  let extraClassDeclaration = [{
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ThreadingMode threading, FunctionForm form,
                         ArrayRef<NamedAttribute> attrs = {});

    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def VC4_ReturnOp : VC4_Op<"return", [
    HasParent<"FuncOp">, ReturnLike, Terminator
  ]> {
  let summary = "VC4 structured return";
  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, ValueRange{});
  }]>];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

def VC4_BuiltinOp : VC4_Op<"builtin", [Pure, HasParent<"FuncOp">]> {
  let summary = "VC4 builtin materialization";
  let arguments = (ins VC4_BuiltinKindAttr:$kind);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def VC4_UniformReadOp : VC4_Op<"uniform.read", [
    HasParent<"FuncOp">, MemoryEffects<[MemRead]>
  ]> {
  let summary = "Read one element from the uniform stream";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4_UniformSeekOp : VC4_Op<"uniform.seek", [
    HasParent<"FuncOp">, MemoryEffects<[MemWrite]>
  ]> {
  let summary = "Reset or adjust the uniform stream pointer";
  let arguments = (ins
    AnyType:$offset,
    DefaultValuedAttr<BoolAttr, "false">:$relative
  );
  let assemblyFormat = "$offset attr-dict `:` type($offset)";
  let hasVerifier = 1;
}

def VC4_MovOp : VC4_Op<"mov", [
    Pure, SameOperandsAndResultType, HasParent<"FuncOp">
  ]> {
  let summary = "Structured identity move";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4_ReadOp : VC4_Op<"read", [
    SameOperandsAndResultType, HasParent<"FuncOp">
  ]> {
  let summary = "Structured vc4asm-style pseudo read";
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4_ALUAddOp : VC4_Op<"alu.add", [HasParent<"FuncOp">]> {
  let summary = "Structured VC4 ADD-pipe operation";
  let arguments = (ins
    Variadic<AnyType>:$operands,
    VC4_AddOpcodeAttr:$op,
    VC4_CondAttr:$cond,
    OptionalAttr<UnitAttr>:$set_flags
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$op $operands attr-dict `:` functional-type($operands, results)";
  let hasVerifier = 1;
}

def VC4_ALUMulOp : VC4_Op<"alu.mul", [HasParent<"FuncOp">]> {
  let summary = "Structured VC4 MUL-pipe operation";
  let arguments = (ins
    Variadic<AnyType>:$operands,
    VC4_MulOpcodeAttr:$op,
    VC4_CondAttr:$cond,
    OptionalAttr<UnitAttr>:$set_flags
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$op $operands attr-dict `:` functional-type($operands, results)";
  let hasVerifier = 1;
}

def VC4_LoadImmOp : VC4_Op<"load_imm", [HasParent<"FuncOp">]> {
  let summary = "Structured VC4 load-immediate materialization";
  let arguments = (ins
    VC4_LoadImmModeAttr:$mode,
    OptionalAttr<AnyAttr>:$value
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4_PackOp : VC4_Op<"pack", [HasParent<"FuncOp">]> {
  let summary = "Structured VC4 pack operation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<VC4_RegfileAPackModeAttr>:$regfile_a_mode,
    OptionalAttr<VC4_MulPackModeAttr>:$mul_mode
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def VC4_UnpackOp : VC4_Op<"unpack", [HasParent<"FuncOp">]> {
  let summary = "Structured VC4 unpack operation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<VC4_RegfileAUnpackModeAttr>:$regfile_a_mode,
    OptionalAttr<VC4_R4UnpackModeAttr>:$r4_mode
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def VC4_RotateOp : VC4_Op<"rotate", [HasParent<"FuncOp">]> {
  let summary = "Structured VC4 horizontal rotate";
  let arguments = (ins
    AnyType:$input,
    Optional<AnySignlessIntegerOrIndex>:$amount,
    OptionalAttr<I32Attr>:$immediate
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def VC4_TMUDescriptorOp : VC4_Op<"tmu.descriptor", [
    Pure, HasParent<"FuncOp">
  ]> {
  let summary = "Structured TMU descriptor construction";
  let arguments = (ins
    VC4_TMUModeAttr:$mode,
    OptionalAttr<I32Attr>:$base,
    OptionalAttr<VC4_TextureTypeAttr>:$texture_type,
    OptionalAttr<I32Attr>:$mip_levels,
    OptionalAttr<I32Attr>:$width,
    OptionalAttr<I32Attr>:$height,
    OptionalAttr<VC4_MagFilterAttr>:$mag_filter,
    OptionalAttr<VC4_MinFilterAttr>:$min_filter,
    OptionalAttr<VC4_WrapModeAttr>:$wrap_s,
    OptionalAttr<VC4_WrapModeAttr>:$wrap_t,
    OptionalAttr<BoolAttr>:$flip_y,
    OptionalAttr<I32Attr>:$cube_map_stride,
    OptionalAttr<DictionaryAttr>:$child_image_fields,
    OptionalAttr<DictionaryAttr>:$bias_flags
  );
  let results = (outs VC4_TMUDescType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4_TMURequestOp : VC4_Op<"tmu.request", [HasParent<"FuncOp">]> {
  let summary = "Structured TMU request enqueue";
  let arguments = (ins
    Variadic<AnyType>:$operands,
    VC4_TMUUnitAttr:$unit
  );
  let results = (outs Optional<VC4_AsyncTokenType>:$token);
  let hasVerifier = 1;
}

def VC4_TMUReadOp : VC4_Op<"tmu.read", [HasParent<"FuncOp">]> {
  let summary = "Structured TMU receive readback";
  let arguments = (ins
    Optional<VC4_AsyncTokenType>:$token,
    VC4_TMUUnitAttr:$unit,
    VC4_TMUReadPartAttr:$part
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def VC4_TMUNoSwapOp : VC4_Op<"tmu.noswap", [HasParent<"FuncOp">]> {
  let summary = "Structured TMU automatic swap control";
  let arguments = (ins
    Optional<AnySignlessInteger>:$value,
    OptionalAttr<BoolAttr>:$disable
  );
  let results = (outs);
  let hasVerifier = 1;
}

def VC4_SFUIssueOp : VC4_Op<"sfu.issue", [HasParent<"FuncOp">]> {
  let summary = "Structured SFU issue";
  let arguments = (ins
    AnyType:$input,
    VC4_SFUKindAttr:$kind
  );
  let assemblyFormat = "$kind $input attr-dict `:` type($input)";
  let hasVerifier = 1;
}

def VC4_SFUReadOp : VC4_Op<"sfu.read", [HasParent<"FuncOp">]> {
  let summary = "Structured SFU readback";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4_VPMDescriptorOp : VC4_Op<"vpm.desc", [
    Pure, HasParent<"FuncOp">
  ]> {
  let summary = "Structured VPM descriptor construction";
  let arguments = (ins
    VC4_VPMDescKindAttr:$kind,
    VC4_VPMOrientationAttr:$orientation,
    VC4_VPMLaneModeAttr:$lane_mode,
    VC4_VPMElemWidthAttr:$elem_width,
    OptionalAttr<I32Attr>:$addr,
    OptionalAttr<I32Attr>:$stride,
    OptionalAttr<I32Attr>:$num_vectors
  );
  let results = (outs VC4_VPMDescType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4_VPMReadOp : VC4_Op<"vpm.read", [HasParent<"FuncOp">]> {
  let summary = "Structured VPM read";
  let arguments = (ins VC4_VPMDescType:$descriptor);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$descriptor attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def VC4_VPMWriteOp : VC4_Op<"vpm.write", [HasParent<"FuncOp">]> {
  let summary = "Structured VPM write";
  let arguments = (ins
    VC4_VPMDescType:$descriptor,
    AnyType:$value
  );
  let assemblyFormat = "$descriptor `,` $value attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def VC4_DMADescriptorOp : VC4_Op<"dma.desc", [
    Pure, HasParent<"FuncOp">
  ]> {
  let summary = "Structured DMA descriptor construction";
  let arguments = (ins
    VC4_DMADescKindAttr:$kind,
    VC4_DMABlockModeAttr:$block_mode,
    VC4_DMAOrientationAttr:$orientation,
    VC4_DMAElemWidthAttr:$elem_width,
    OptionalAttr<I32Attr>:$start_offset,
    OptionalAttr<I32Attr>:$mpitch,
    OptionalAttr<I32Attr>:$vpitch,
    OptionalAttr<I32Attr>:$nrows,
    OptionalAttr<I32Attr>:$rowlen,
    OptionalAttr<I32Attr>:$units,
    OptionalAttr<I32Attr>:$depth,
    OptionalAttr<I32Attr>:$vpm_base,
    OptionalAttr<I32Attr>:$stride,
    OptionalAttr<I32Attr>:$extended_stride
  );
  let results = (outs VC4_DMADescType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4_DMAStartOp : VC4_Op<"dma.start", [HasParent<"FuncOp">]> {
  let summary = "Structured DMA start";
  let arguments = (ins
    VC4_DMADescType:$descriptor,
    AnyType:$base
  );
  let results = (outs Optional<VC4_AsyncTokenType>:$token);
  let hasVerifier = 1;
}

def VC4_DMAStatusOp : VC4_Op<"dma.status", [HasParent<"FuncOp">]> {
  let summary = "Structured DMA status poll";
  let arguments = (ins VC4_DMADescKindAttr:$kind);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$kind attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4_DMAWaitOp : VC4_Op<"dma.wait", [HasParent<"FuncOp">]> {
  let summary = "Structured DMA completion wait";
  let arguments = (ins
    Optional<VC4_AsyncTokenType>:$token,
    OptionalAttr<VC4_DMADescKindAttr>:$kind
  );
  let results = (outs);
  let hasVerifier = 1;
}

def VC4_MutexOp : VC4_Op<"mutex", [HasParent<"FuncOp">]> {
  let summary = "Structured hardware mutex action";
  let arguments = (ins VC4_MutexModeAttr:$mode);
  let assemblyFormat = "$mode attr-dict";
  let hasVerifier = 1;
}

def VC4_SemaphoreOp : VC4_Op<"semaphore", [HasParent<"FuncOp">]> {
  let summary = "Structured hardware semaphore action";
  let arguments = (ins
    VC4_SemaphoreModeAttr:$mode,
    I32Attr:$id
  );
  let assemblyFormat = "$mode attr-dict";
  let hasVerifier = 1;
}

def VC4_HostInterruptOp : VC4_Op<"host_interrupt", [HasParent<"FuncOp">]> {
  let summary = "Structured host interrupt trigger";
  let results = (outs);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def VC4_ThreadSwitchOp : VC4_Op<"thread_switch", [HasParent<"FuncOp">]> {
  let summary = "Structured cooperative thread switch";
  let arguments = (ins VC4_ThreadSwitchModeAttr:$mode);
  let assemblyFormat = "$mode attr-dict";
  let hasVerifier = 1;
}

def VC4_ProgramEndOp : VC4_Op<"program_end", [HasParent<"FuncOp">]> {
  let summary = "Structured program end signal";
  let results = (outs);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def VC4_AsyncWaitOp : VC4_Op<"async.wait", [HasParent<"FuncOp">]> {
  let summary = "Structured async token join";
  let arguments = (ins Variadic<VC4_AsyncTokenType>:$tokens);
  let results = (outs);
  let assemblyFormat = "attr-dict ($tokens^ `:` type($tokens))?";
  let hasVerifier = 1;
}

def VC4_CFBranchOp : VC4_Op<"cf.branch", [
    DeclareOpInterfaceMethods<BranchOpInterface>, HasParent<"FuncOp">,
    Pure, Terminator
  ]> {
  let summary = "Structured hardware-facing flag branch";
  let arguments = (ins VC4_BranchCondAttr:$cond);
  let successors = (successor VariadicSuccessor<AnySuccessor>:$successors);
  let hasVerifier = 1;
}

def VC4_EnqueueQPUOp : VC4_Op<"enqueue_qpu", [HasParent<"FuncOp">]> {
  let summary = "Structured QPU scheduler enqueue";
  let arguments = (ins
    Variadic<AnyType>:$operands,
    FlatSymbolRefAttr:$entry
  );
  let results = (outs Optional<VC4_AsyncTokenType>:$token);
  let hasVerifier = 1;
}

def VC4_ReserveQPUOp : VC4_Op<"reserve_qpu", [HasParent<"FuncOp">]> {
  let summary = "Structured QPU reservation control";
  let arguments = (ins I32Attr:$mask);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def VC4_V3DQueryOp : VC4_Op<"v3d.query", [HasParent<"FuncOp">]> {
  let summary = "Structured V3D system query";
  let arguments = (ins
    Variadic<AnyType>:$selectors,
    VC4_V3DQueryKindAttr:$kind
  );
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
}

def VC4_V3DConfigureOp : VC4_Op<"v3d.configure", [HasParent<"FuncOp">]> {
  let summary = "Structured V3D system configure";
  let arguments = (ins
    Variadic<AnyType>:$payloads,
    VC4_V3DConfigureKindAttr:$kind
  );
  let results = (outs);
  let hasVerifier = 1;
}

#endif // VC4_STRUCTURED_OPS_TD
