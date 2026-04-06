#include "vc4/Target/VC4Asm/VC4Asm.h"

#include "mlir/InitAllDialects.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "vc4/Dialect/VC4/VC4Dialect.h"

namespace mlir::vc4 {

void registerToVC4AsmTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-vc4asm", "Translate MLIR to placeholder vc4asm output",
      [](Operation *op, llvm::raw_ostream &output) {
        output << "; vc4asm translation stub\n";
        op->print(output, OpPrintingFlags().assumeVerified());
        output << "\n";
        return success();
      },
      [](DialectRegistry &registry) {
        registerAllDialects(registry);
        registry.insert<VC4Dialect>();
      });
}

} // namespace mlir::vc4
