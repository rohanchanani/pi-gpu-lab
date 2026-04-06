include "mlir/IR/DialectBase.td"

def VC4_Dialect : Dialect {
  let name = "vc4";
  let cppNamespace = "::mlir::vc4";
  let summary = "VC4 staged-memory dialect";
  let description = [{
    The vc4 dialect models VideoCore IV machine structure at the staged-memory
    level, above raw vc4asm syntax.
  }];
}
