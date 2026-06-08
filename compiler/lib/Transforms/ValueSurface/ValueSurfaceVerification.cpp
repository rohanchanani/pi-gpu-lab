//===- ValueSurfaceVerification.cpp - VC4 value surface verifier ----------===//

#include "vc4/Transforms/ValueSurface/ValueSurfacePasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

using namespace mlir;

namespace {

static StringRef getDialectNamespace(Operation *op) {
  return op->getName().getStringRef().split('.').first;
}

static bool isForbiddenTargetDialect(StringRef dialect) {
  return dialect == "vc4kernel" || dialect == "ssavc4" || dialect == "vc4";
}

static bool isForbiddenProducerDialect(StringRef dialect) {
  return dialect == "tt" || dialect == "ttg" || dialect == "gpu" ||
         dialect == "linalg" || dialect == "nvgpu" || dialect == "nvvm" ||
         dialect == "rocdl" || dialect == "spirv" || dialect == "iree" ||
         dialect == "stablehlo" || dialect == "mhlo";
}

static bool isLegalVC4ValueOp(StringRef name) {
  return name == "vc4value.program_id" || name == "vc4value.num_programs";
}

struct VerifyValueSurfacePass
    : public PassWrapper<VerifyValueSurfacePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyValueSurfacePass)

  StringRef getArgument() const final { return "vc4-verify-value-surface"; }

  StringRef getDescription() const final {
    return "verify VC4 standard value-surface boundary admissibility";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    unsigned kernelCount = 0;
    bool sawError = false;

    module.walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();
      StringRef dialect = getDialectNamespace(op);

      if (opName == "func.func" && op->hasAttr("vc4value.kernel"))
        ++kernelCount;

      if (isForbiddenTargetDialect(dialect)) {
        op->emitError() << "target/lower-half dialect is not legal in the "
                        << "VC4 value surface: " << dialect;
        sawError = true;
        return WalkResult::advance();
      }

      if (isForbiddenProducerDialect(dialect)) {
        op->emitError() << "producer dialect is not legal in the VC4 value "
                        << "surface: " << dialect;
        sawError = true;
        return WalkResult::advance();
      }

      if (dialect == "vc4value" && !isLegalVC4ValueOp(opName)) {
        op->emitError()
            << "only vc4value.program_id and vc4value.num_programs are "
            << "legal in Phase 3; got " << opName;
        sawError = true;
      }

      return WalkResult::advance();
    });

    if (kernelCount == 0) {
      module.emitError()
          << "expected at least one func.func marked with vc4value.kernel";
      sawError = true;
    }

    if (sawError)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vc4::createVerifyValueSurfacePass() {
  return std::make_unique<VerifyValueSurfacePass>();
}

void mlir::vc4::registerValueSurfacePasses() {
  // The file-scope PassRegistration below installs
  // --vc4-verify-value-surface when this library is linked into vc4-opt.
}

static PassRegistration<VerifyValueSurfacePass> registerVerifyValueSurfacePass;
