include "mlir/IR/DialectBase.td"

def VC4_Dialect : Dialect {
  let name = "vc4";
  let cppNamespace = "::mlir::vc4";
  let summary = "VC4 lowered kernel dialect";
  let description = [{
    The vc4 dialect models the lowered kernel-side contract for the VideoCore IV
    backend above raw vc4asm syntax. It makes execution builtins, logical
    uniform positions, and staged-memory movement explicit while keeping staging
    locations abstract so later emission can choose physical VPM rows.
  }];
}
