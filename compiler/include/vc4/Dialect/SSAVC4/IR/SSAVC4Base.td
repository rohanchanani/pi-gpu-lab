#ifndef SSAVC4_BASE_TD
#define SSAVC4_BASE_TD

include "mlir/IR/OpBase.td"

// SSAVC4 is the pre-register-allocation, pre-scheduling VC4 machine SSA IR.
def SSAVC4_Dialect : Dialect {
  let name = "ssavc4";
  let cppNamespace = "::mlir::ssavc4";
  let dependentDialects = ["::mlir::vc4::VC4Dialect"];
  let summary = "SSA VideoCore IV target machine IR";
  let description = [{
    SSAVC4 models low-level VideoCore IV QPU/VPM/TMU/VDW/SFU semantics with
    SSA values and explicit effect ordering before lowering to the scheduled
    `vc4` artifact sink.  It intentionally does not expose final physical
    register addresses, final instruction order, branch immediates, or branch
    delay-slot layout.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // SSAVC4_BASE_TD
