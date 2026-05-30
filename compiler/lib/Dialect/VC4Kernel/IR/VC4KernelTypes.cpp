//===- VC4KernelTypes.cpp - VC4Kernel dialect types -----------------------===//

#include "vc4/Dialect/VC4Kernel/IR/VC4KernelTypes.h"

#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::vc4kernel;

LogicalResult PredType::verify(function_ref<InFlightDiagnostic()> emitError,
                               unsigned width) {
  if (width == 16)
    return success();
  return emitError() << "!vc4kernel.pred width must be 16";
}

bool mlir::vc4kernel::isVC4KernelScalarType(Type type) {
  return type &&
         (type.isInteger(1) || type.isSignlessInteger(32) || type.isF32());
}

static bool isVector16Of(Type type, llvm::function_ref<bool(Type)> pred) {
  auto vectorType = llvm::dyn_cast<VectorType>(type);
  if (!vectorType || vectorType.getRank() != 1 ||
      vectorType.getDimSize(0) != 16)
    return false;
  return pred(vectorType.getElementType());
}

bool mlir::vc4kernel::isVC4KernelVector16I32Type(Type type) {
  return isVector16Of(type, [](Type element) {
    return element.isSignlessInteger(32);
  });
}

bool mlir::vc4kernel::isVC4KernelVector16F32Type(Type type) {
  return isVector16Of(type, [](Type element) { return element.isF32(); });
}

bool mlir::vc4kernel::isVC4KernelVector16DataType(Type type) {
  return isVC4KernelVector16I32Type(type) ||
         isVC4KernelVector16F32Type(type);
}

bool mlir::vc4kernel::isVC4KernelPredType(Type type) {
  auto pred = llvm::dyn_cast_if_present<PredType>(type);
  return pred && pred.getWidth() == 16;
}

bool mlir::vc4kernel::isVC4KernelVPMTileType(Type type) {
  return type && llvm::isa<VPMTileType>(type);
}

bool mlir::vc4kernel::isLegalVC4KernelType(Type type) {
  return isVC4KernelScalarType(type) || isVC4KernelVector16DataType(type) ||
         isVC4KernelPredType(type) || isVC4KernelVPMTileType(type);
}
