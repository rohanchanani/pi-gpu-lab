include "mlir/IR/DialectBase.td"

def VC4_Dialect : Dialect {
  let name = "vc4";
  let cppNamespace = "::mlir::vc4";
  let summary = "VC4 compute/QPU dialect";
  let description = [{
    Minimal dialect shell for the standalone VC4 MLIR project.
  }];
}
