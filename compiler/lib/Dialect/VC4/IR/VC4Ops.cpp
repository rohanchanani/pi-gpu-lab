#include "vc4/Dialect/VC4/VC4Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::vc4;

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4/VC4Ops.cpp.inc"
