#ifndef VC4VALUE_OPS_TD
#define VC4VALUE_OPS_TD

include "vc4/Dialect/VC4Value/IR/VC4ValueDialect.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VC4Value_Op<string mnemonic, list<Trait> traits = []>
    : Op<VC4Value_Dialect, mnemonic, traits>;

class VC4Value_PureOp<string mnemonic, list<Trait> traits = []>
    : VC4Value_Op<mnemonic, [Pure] # traits>;

def VC4Value_ProgramIdOp : VC4Value_PureOp<"program_id"> {
  let summary = "logical value-layer program id for a launch-grid axis";
  let arguments = (ins I32Attr:$axis);
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Value_NumProgramsOp : VC4Value_PureOp<"num_programs"> {
  let summary = "logical value-layer launch-grid extent for an axis";
  let arguments = (ins I32Attr:$axis);
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

#endif // VC4VALUE_OPS_TD
