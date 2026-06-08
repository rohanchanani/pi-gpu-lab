//===- VC4ValueOps.cpp - VC4Value operation implementation ----------------===//

#include "vc4/Dialect/VC4Value/IR/VC4ValueOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::vc4value;

namespace {

static LogicalResult verifyLaunchAxisOp(Operation *op) {
  if (op->getNumResults() != 1 || !op->getResult(0).getType().isIndex())
    return op->emitOpError("result type must be index");

  auto axisAttr = llvm::dyn_cast_if_present<IntegerAttr>(op->getAttr("axis"));
  if (!axisAttr)
    return op->emitOpError("axis must be an i32 integer attribute");
  if (!axisAttr.getType().isSignlessInteger(32))
    return op->emitOpError("axis must be an i32 integer attribute");

  int64_t axis = axisAttr.getInt();
  if (axis < 0 || axis > 2)
    return op->emitOpError("axis must be 0, 1, or 2");
  return success();
}

} // namespace

LogicalResult ProgramIdOp::verify() {
  return verifyLaunchAxisOp(getOperation());
}

LogicalResult NumProgramsOp::verify() {
  return verifyLaunchAxisOp(getOperation());
}

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4Value/IR/VC4ValueOps.cpp.inc"
