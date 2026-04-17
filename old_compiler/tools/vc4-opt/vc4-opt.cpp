#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "vc4/Conversion/GPUToVC4/GPUToVC4.h"
#include "vc4/Dialect/VC4/VC4Dialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::vc4::VC4Dialect>();

  mlir::registerAllPasses();
  mlir::vc4::registerGPUToVC4Passes();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "VC4 optimizer driver\n", registry));
}
