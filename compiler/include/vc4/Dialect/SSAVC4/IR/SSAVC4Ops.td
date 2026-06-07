#ifndef SSAVC4_OPS_TD
#define SSAVC4_OPS_TD

include "vc4/Dialect/SSAVC4/IR/SSAVC4Dialect.td"
include "vc4/Dialect/SSAVC4/IR/SSAVC4Types.td"
include "vc4/Dialect/SSAVC4/IR/SSAVC4Attrs.td"
include "vc4/Dialect/VC4/IR/VC4Attrs.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SSAVC4_Op<string mnemonic, list<Trait> traits = []>
    : Op<SSAVC4_Dialect, mnemonic, traits>;

class SSAVC4_PureOp<string mnemonic, list<Trait> traits = []>
    : SSAVC4_Op<mnemonic, [Pure] # traits>;

def SSAVC4_ModuleOp : SSAVC4_Op<"module", [
    IsolatedFromAbove, NoRegionArguments, Symbol, SymbolTable, NoTerminator
  ]> {
  let summary = "SSAVC4 target module";
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def SSAVC4_FuncOp : SSAVC4_Op<"func", [
    IsolatedFromAbove, NoRegionArguments, Symbol
  ]> {
  let summary = "SSAVC4 pre-scheduled QPU function";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<UnitAttr>:$kernel,
    OptionalAttr<VC4_ThreadingModeAttr>:$threading
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name `(` `)` attr-dict-with-keyword $body";
}

def SSAVC4_ThreadEndOp : SSAVC4_Op<"thread_end", [Terminator]> {
  let summary = "SSAVC4 QPU thread termination marker";
  let assemblyFormat = "attr-dict";
}

def SSAVC4_LoadImmOp : SSAVC4_PureOp<"load_imm"> {
  let summary = "SSAVC4 pure immediate value";
  let arguments = (ins
    VC4_LoadImmModeAttr:$mode,
    OptionalAttr<AnyAttr>:$value,
    OptionalAttr<AnyAttr>:$values
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$mode attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def SSAVC4_ElementNumberOp : SSAVC4_PureOp<"element_number"> {
  let summary = "SSAVC4 vector lane id value";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def SSAVC4_UniformReadOp : SSAVC4_Op<"uniform.read"> {
  let summary = "SSAVC4 sequential uniform stream read";
  let description = [{
    Reads the next physical QPU uniform word. The optional index documents the
    expected launch ABI slot; ordering is defined by program order so lowering
    remains independent of public kernel names.
  }];
  let arguments = (ins I32Attr:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$index attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def SSAVC4_SplatOp : SSAVC4_PureOp<"splat"> {
  let summary = "SSAVC4 scalar-to-vector splat";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def SSAVC4_MovOp : SSAVC4_PureOp<"mov"> {
  let summary = "SSAVC4 pure SSA move";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def SSAVC4_CondSelectOp : SSAVC4_PureOp<"cond_select"> {
  let summary = "SSAVC4 per-lane conditional value select";
  let description = [{
    Selects true_value for SIMD lanes whose current QPU flags satisfy cond,
    and false_value otherwise.  This models the natural VC4 idiom:

      set flags per SIMD lane
      write the false value unconditionally
      conditionally overwrite lanes with the true value

    The flags operand must be produced by ssavc4.make_flags and consumed once,
    preserving the lower-half invariant that live flag state is explicit and
    short-lived.
  }];
  let arguments = (ins
    SSAVC4_Flags:$flags,
    AnyType:$true_value,
    AnyType:$false_value,
    VC4_CondAttr:$cond
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $flags `,` $true_value `,` $false_value attr-dict
    `:` type($flags) `,` type($true_value) `,` type($false_value)
    `->` type($result)
  }];
  let hasVerifier = 1;
}

def SSAVC4_ALUAddOp : SSAVC4_PureOp<"alu.add"> {
  let summary = "SSAVC4 ADD-pipe ALU operation before scheduling";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    VC4_AddOpcodeAttr:$opcode
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $result)";
  let hasVerifier = 1;
}

def SSAVC4_ALUMulOp : SSAVC4_PureOp<"alu.mul"> {
  let summary = "SSAVC4 MUL-pipe ALU operation before scheduling";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    VC4_MulOpcodeAttr:$opcode
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $result)";
  let hasVerifier = 1;
}

def SSAVC4_SFUOp : SSAVC4_Op<"sfu"> {
  let summary = "SSAVC4 ordered VC4 special-functions unit operation";
  let description = [{
    Issues one finite-domain approximate VC4 SFU operation and returns the
    result delivered through accumulator r4 after the architectural two-slot
    hazard window.  The operation preserves the approximate policy and domain
    contract from VC4Kernel instead of becoming a raw scheduled register write.
  }];
  let arguments = (ins
    AnyType:$input,
    SSAVC4_SFUKindAttr:$kind,
    SSAVC4_FPMathPolicyAttr:$fp_policy,
    SSAVC4_FPDomainAttr:$domain
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def SSAVC4_PackOp : SSAVC4_PureOp<"pack"> {
  let summary = "SSAVC4 pack value transform";
  let arguments = (ins AnyType:$input, OptionalAttr<AnyAttr>:$mode);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def SSAVC4_UnpackOp : SSAVC4_PureOp<"unpack"> {
  let summary = "SSAVC4 unpack value transform";
  let arguments = (ins AnyType:$input, OptionalAttr<AnyAttr>:$mode);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def SSAVC4_RotateOp : SSAVC4_PureOp<"rotate"> {
  let summary = "SSAVC4 vector lane rotate";
  let arguments = (ins AnyType:$input, I32Attr:$amount);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def SSAVC4_MakeFlagsOp : SSAVC4_PureOp<"make_flags"> {
  let summary = "SSAVC4 pure flag pseudo-value producer";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    SSAVC4_FlagKindAttr:$kind
  );
  let results = (outs SSAVC4_Flags:$result);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $result)";
  let hasVerifier = 1;
}



def SSAVC4_BranchOp : SSAVC4_Op<"br", [Terminator]> {
  let summary = "SSAVC4 unconditional block branch";
  let arguments = (ins Variadic<AnyType>:$targetOperands);
  let successors = (successor AnySuccessor:$target);
  let assemblyFormat = "$target (`(` $targetOperands^ `:` type($targetOperands) `)`)? attr-dict";
  let hasVerifier = 1;
}

def SSAVC4_CondBranchOp : SSAVC4_Op<"cond_br", [
    AttrSizedOperandSegments, Terminator
  ]> {
  let summary = "SSAVC4 conditional branch consuming a flag pseudo-value";
  let arguments = (ins
    SSAVC4_Flags:$flags,
    Variadic<AnyType>:$trueDestOperands,
    Variadic<AnyType>:$falseDestOperands,
    VC4_BranchCondAttr:$cond
  );
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = [{
    $flags `,`
    $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
    $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
    attr-dict `:` type($flags)
  }];
  let hasVerifier = 1;
}




def SSAVC4_SemaAcquireOp : SSAVC4_Op<"sema.acquire"> {
  let summary = "SSAVC4 effectful semaphore acquire";
  let description = [{
    Acquires a target hardware semaphore.  M3 v1 requires the semaphore
    operand to resolve to a compile-time i32 id during SSAVC4-to-VC4 lowering.
    Dynamic runtime semaphore ids are reserved for a later spill/resource slice.
  }];
  let arguments = (ins AnyType:$semaphore);
  let assemblyFormat = "$semaphore attr-dict `:` type($semaphore)";
  let hasVerifier = 1;
}

def SSAVC4_SemaReleaseOp : SSAVC4_Op<"sema.release"> {
  let summary = "SSAVC4 effectful semaphore release";
  let description = [{
    Releases a target hardware semaphore.  M3 v1 requires the semaphore
    operand to resolve to a compile-time i32 id during SSAVC4-to-VC4 lowering.
  }];
  let arguments = (ins AnyType:$semaphore);
  let assemblyFormat = "$semaphore attr-dict `:` type($semaphore)";
  let hasVerifier = 1;
}

def SSAVC4_BarrierOp : SSAVC4_Op<"barrier"> {
  let summary = "SSAVC4 cooperative-block barrier";
  let description = [{
    Models a block-scoped cooperative barrier backed by the M2 scheduled
    semaphore resource contract.  The operation is legal only with
    vc4.resource metadata declaring cooperative_block scheduling, barrier use,
    a positive warps_per_block, and enough hardware
    semaphores for the four offset attributes.
  }];
  let arguments = (ins
    Variadic<AnyType>:$operands,
    I32Attr:$arrive_offset,
    I32Attr:$go_offset,
    I32Attr:$depart_offset,
    I32Attr:$reset_offset
  );
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

def SSAVC4_TMURequestOp : SSAVC4_Op<"tmu.request"> {
  let summary = "SSAVC4 effectful direct TMU request";
  let description = [{
    Issues a direct TMU memory request.  M3 v1 supports tmu0/direct/raw32
    request/read pairs ordered by the explicit async token produced here and
    consumed by ssavc4.tmu.read.  The operation is intentionally not Pure.
  }];
  let arguments = (ins
    AnyType:$address,
    StrAttr:$unit,
    StrAttr:$mode
  );
  let results = (outs AnyType:$token);
  let assemblyFormat = "$address attr-dict `:` type($address) `->` type($token)";
}

def SSAVC4_TMUReadOp : SSAVC4_Op<"tmu.read"> {
  let summary = "SSAVC4 effectful direct TMU read";
  let description = [{
    Receives the value for a prior direct TMU request.  The input token ties
    the read to the request and prevents request/read reordering in SSAVC4.
    M3 v1 supports unit = "tmu0" and part = "raw32".
  }];
  let arguments = (ins
    AnyType:$token,
    StrAttr:$unit,
    StrAttr:$part
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$token attr-dict `:` type($token) `->` type($result)";
}


def SSAVC4_VPMWriteOp : SSAVC4_Op<"vpm.write"> {
  let summary = "SSAVC4 effectful shared VPM vector write";
  let description = [{
    Writes one 16-lane vector value to VPM through QPU register-space VPM
    setup/write traffic.  The typed mode attrs describe the hardware-derived
    32-bit VPM setup word. P9 accepts w32/none plus hardware-proven w8/w16
    packed and laned QPU VPM modes.
  }];
  let arguments = (ins
    AnyType:$row,
    AnyType:$value,
    SSAVC4_VPMOrientationAttr:$orientation,
    SSAVC4_VPMElemWidthAttr:$width,
    SSAVC4_VPMSubwordAttr:$subword,
    I32Attr:$x,
    I32Attr:$stride,
    I32Attr:$lanes
  );
  let assemblyFormat = "$row `,` $value attr-dict `:` type($row) `,` type($value)";
  let hasVerifier = 1;
}

def SSAVC4_VPMReadOp : SSAVC4_Op<"vpm.read"> {
  let summary = "SSAVC4 effectful shared VPM vector read";
  let description = [{
    Reads one 16-lane vector value from VPM through QPU register-space VPM
    setup/read traffic.  The typed mode attrs describe the hardware-derived
    32-bit VPM setup word. P9 accepts w32/none plus hardware-proven w8/w16
    packed and laned QPU VPM modes.
  }];
  let arguments = (ins
    AnyType:$row,
    SSAVC4_VPMOrientationAttr:$orientation,
    SSAVC4_VPMElemWidthAttr:$width,
    SSAVC4_VPMSubwordAttr:$subword,
    I32Attr:$x,
    I32Attr:$stride,
    I32Attr:$lanes
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$row attr-dict `:` type($row) `->` type($result)";
  let hasVerifier = 1;
}



def SSAVC4_VDRLoadOp : SSAVC4_Op<"vdr.load"> {
  let summary = "SSAVC4 effectful global-to-VPM load through VDR/VCD";
  let description = [{
    Starts and waits for a lower-half VideoCore IV VDR/VCD DMA load from a
    global-memory base address into the shared VPM window.  This is the
    hardware-natural global-to-shared path that kernel planning
    can target for regular 32-bit tile loads.  The operation is intentionally
    not Pure: lowering owns the concrete VPMVCD_RD_SETUP, VPM_LD_ADDR, wait,
    and optional mutex serialization sequence.  M5 executable semantics are
    32-bit-only, so width must be w32 and subword must be none.
  }];
  let arguments = (ins
    AnyType:$address,
    AnyType:$vpm_base_row,
    SSAVC4_VPMOrientationAttr:$orientation,
    SSAVC4_VPMElemWidthAttr:$width,
    SSAVC4_VPMSubwordAttr:$subword,
    I32Attr:$row_len,
    I32Attr:$nrows,
    I32Attr:$memory_pitch_bytes,
    I32Attr:$vpm_x,
    I32Attr:$vpm_pitch,
    OptionalAttr<StrAttr>:$serialize
  );
  let assemblyFormat = "$address `,` $vpm_base_row attr-dict `:` type($address) `,` type($vpm_base_row)";
  let hasVerifier = 1;
}

def SSAVC4_VDRLoadRectDynamicOp : SSAVC4_Op<"vdr.load_rect.dynamic", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "SSAVC4 dynamic rectangular global-to-VPM load";
  let description = [{
    Represents a semantic dynamic rectangular VDR/VCD transfer from global
    memory to VPM.  The maximum tile shape is static, while active rows,
    active columns, and memory pitch are scalar SSA operands.  Inactive or
    out-of-rectangle destination VPM elements are zero-filled.
  }];
  let arguments = (ins
    AnyType:$address,
    AnyType:$vpm_base_row,
    AnyType:$active_rows,
    AnyType:$active_cols,
    AnyType:$memory_pitch_bytes,
    I32Attr:$max_rows,
    I32Attr:$max_cols,
    I32Attr:$elem_bytes,
    SSAVC4_VPMOrientationAttr:$orientation,
    SSAVC4_VPMElemWidthAttr:$width,
    SSAVC4_VPMSubwordAttr:$subword,
    I32Attr:$dst_x,
    I32Attr:$vpm_pitch,
    BoolAttr:$zero_fill,
    OptionalAttr<StrAttr>:$serialize
  );
  let assemblyFormat = "$address `,` $vpm_base_row `,` $active_rows `,` $active_cols `,` $memory_pitch_bytes attr-dict `:` type($address) `,` type($vpm_base_row) `,` type($active_rows) `,` type($active_cols) `,` type($memory_pitch_bytes)";
  let hasVerifier = 1;
}

def SSAVC4_VDWStoreOp : SSAVC4_Op<"vdw.store", [AttrSizedOperandSegments]> {
  let summary = "SSAVC4 effectful global vector store through VPM/VDW";
  let description = [{
    Stages one 16-lane vector value through VPM and commits it through the
    VideoCore IV VDW global-store path during SSAVC4-to-scheduled-VC4 lowering.
    The operation is intentionally not Pure: the lowering owns the concrete
    VPM/VDW setup, waits, and optional mutex serialization.
  }];
  let arguments = (ins
    AnyType:$address,
    AnyType:$value,
    Optional<AnyType>:$active_lanes_value,
    Optional<AnyType>:$vpm_row_value,
    SSAVC4_VPMElemWidthAttr:$width,
    SSAVC4_VPMSubwordAttr:$subword,
    OptionalAttr<I32Attr>:$active_lanes,
    I32Attr:$vpm_row,
    OptionalAttr<StrAttr>:$serialize
  );
  let assemblyFormat = "$address `,` $value (`,` $active_lanes_value^)? (`,` $vpm_row_value^)? attr-dict `:` type($address) `,` type($value) (`,` type($active_lanes_value)^)? (`,` type($vpm_row_value)^)?";
}

def SSAVC4_VDWStoreVPMOp : SSAVC4_Op<"vdw.store_vpm"> {
  let summary = "SSAVC4 effectful global store from existing VPM coordinates";
  let description = [{
    Commits data already resident in VPM through the VideoCore IV VDW
    global-store path.  Unlike ssavc4.vdw.store, this operation does not stage
    an SSA vector value into a scratch row first; it describes a VDW DMA source
    row/column/orientation in the existing VPM window.

    A dynamic active_lanes_value is a one-row active-depth request.  Lowering
    clamps it to [0, row_len] before programming the VDW descriptor.  Multi-row
    dynamic predicates must be represented as row-decomposed store_vpm ops.
  }];
  let arguments = (ins
    AnyType:$address,
    AnyType:$vpm_y,
    AnyType:$vpm_x,
    Optional<AnyType>:$active_lanes_value,
    SSAVC4_VPMOrientationAttr:$orientation,
    SSAVC4_VPMElemWidthAttr:$width,
    SSAVC4_VPMSubwordAttr:$subword,
    I32Attr:$row_len,
    I32Attr:$nrows,
    I32Attr:$memory_pitch_bytes,
    OptionalAttr<I32Attr>:$active_lanes,
    OptionalAttr<StrAttr>:$serialize
  );
  let assemblyFormat = "$address `,` $vpm_y `,` $vpm_x (`,` $active_lanes_value^)? attr-dict `:` type($address) `,` type($vpm_y) `,` type($vpm_x) (`,` type($active_lanes_value)^)?";
  let hasVerifier = 1;
}

def SSAVC4_VDWStoreRectDynamicOp : SSAVC4_Op<"vdw.store_rect.dynamic", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "SSAVC4 dynamic rectangular VPM-to-global store";
  let description = [{
    Represents a semantic dynamic rectangular VDW transfer from VPM to global
    memory.  The maximum tile shape is static, while active rows, active
    columns, and memory stride are scalar SSA operands.  Inactive or
    out-of-rectangle destination global memory is preserved.
  }];
  let arguments = (ins
    AnyType:$address,
    AnyType:$vpm_source_row,
    AnyType:$active_rows,
    AnyType:$active_cols,
    AnyType:$memory_stride_bytes,
    I32Attr:$max_rows,
    I32Attr:$max_cols,
    I32Attr:$elem_bytes,
    SSAVC4_VPMOrientationAttr:$orientation,
    SSAVC4_VPMElemWidthAttr:$width,
    SSAVC4_VPMSubwordAttr:$subword,
    I32Attr:$src_x,
    I32Attr:$vpm_pitch,
    BoolAttr:$preserve_inactive,
    OptionalAttr<StrAttr>:$serialize
  );
  let assemblyFormat = "$address `,` $vpm_source_row `,` $active_rows `,` $active_cols `,` $memory_stride_bytes attr-dict `:` type($address) `,` type($vpm_source_row) `,` type($active_rows) `,` type($active_cols) `,` type($memory_stride_bytes)";
  let hasVerifier = 1;
}

#endif // SSAVC4_OPS_TD
