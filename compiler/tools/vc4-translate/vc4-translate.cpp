#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "vc4/Target/VC4Asm/VC4Asm.h"

int main(int argc, char **argv) {
  mlir::vc4::registerToVC4AsmTranslation();

  return failed(mlir::mlirTranslateMain(argc, argv, "VC4 translation driver\n"));
}
