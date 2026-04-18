#ifndef VC4_BASE_TD
#define VC4_BASE_TD

include "mlir/IR/DialectBase.td"

def VC4_Dialect : Dialect {
  let name = "vc4";
  let cppNamespace = "::mlir::vc4";
  let summary = "VC4 compute/QPU dialect";
  let description = [{
    Dialect shell for the compute-focused VC4 MLIR target IR.
  }];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // VC4_BASE_TD
