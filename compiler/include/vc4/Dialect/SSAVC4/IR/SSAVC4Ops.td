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
  let successors = (successor AnySuccessor:$target);
  let assemblyFormat = "$target attr-dict";
}

def SSAVC4_CondBranchOp : SSAVC4_Op<"cond_br", [Terminator]> {
  let summary = "SSAVC4 conditional branch consuming a flag pseudo-value";
  let arguments = (ins
    SSAVC4_Flags:$flags,
    VC4_BranchCondAttr:$cond
  );
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = "$flags `,` $trueDest `,` $falseDest attr-dict `:` type($flags)";
}

def SSAVC4_VDWStoreOp : SSAVC4_Op<"vdw.store"> {
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
    I32Attr:$elem_bytes,
    I32Attr:$active_lanes,
    I32Attr:$vpm_row,
    OptionalAttr<StrAttr>:$serialize,
    OptionalAttr<StrAttr>:$lowering_template
  );
  let assemblyFormat = "$address `,` $value attr-dict `:` type($address) `,` type($value)";
}

#endif // SSAVC4_OPS_TD
