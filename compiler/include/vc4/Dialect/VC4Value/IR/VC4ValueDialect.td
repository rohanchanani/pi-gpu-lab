#ifndef VC4VALUE_DIALECT_TD
#define VC4VALUE_DIALECT_TD

include "mlir/IR/OpBase.td"

def VC4Value_Dialect : Dialect {
  let name = "vc4value";
  let cppNamespace = "::mlir::vc4value";
  let summary = "minimal VC4 value-layer launch/policy dialect";
  let description = [{
    vc4value is tiny launch/policy plumbing for the standard MLIR value
    surface above VC4Kernel. It is not VC4Kernel, it is not a tile DSL, it
    must not expose TMU/VDR/VDW/VPM, it must not carry fragment operations,
    and it does not lower directly to VC4.
  }];
}

#endif // VC4VALUE_DIALECT_TD
