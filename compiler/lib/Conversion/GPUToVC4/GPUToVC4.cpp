#include "vc4/Conversion/GPUToVC4/GPUToVC4.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::vc4 {
namespace {

struct ConvertGPUToVC4Pass
    : public PassWrapper<ConvertGPUToVC4Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertGPUToVC4Pass)

  StringRef getArgument() const final { return "convert-gpu-to-vc4"; }
  StringRef getDescription() const final {
    return "Placeholder conversion from GPU-style IR to the VC4 dialect";
  }

  void runOnOperation() final {}
};

} // namespace

std::unique_ptr<Pass> createConvertGPUToVC4Pass() {
  return std::make_unique<ConvertGPUToVC4Pass>();
}

void registerGPUToVC4Passes() {
  PassRegistration<ConvertGPUToVC4Pass>();
}

} // namespace mlir::vc4
