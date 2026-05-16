#ifndef SSAVC4_BASE_TD
#define SSAVC4_BASE_TD

include "mlir/IR/DialectBase.td"

// SSAVC4 is the pre-register-allocation, pre-scheduling machine-SSA
// VideoCore IV dialect.  This scaffold deliberately defines only the dialect
// identity; operation, type, attribute, and lowering semantics arrive in later
// M3 slices.
def SSAVC4_Dialect : Dialect {
  let name = "ssavc4";
  let cppNamespace = "::mlir::ssavc4";
  let summary = "VideoCore IV QPU machine-SSA dialect";
  let description = [{
    SSAVC4 models target-specific VideoCore IV QPU semantics before physical
    register allocation and final scheduling.  It is separate from the locked
    scheduled `vc4` artifact sink.
  }];
}

#endif // SSAVC4_BASE_TD
