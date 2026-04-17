#include "vc4/Dialect/VC4/VC4Dialect.h"

#include "vc4/Dialect/VC4/VC4Ops.h"

using namespace mlir;
using namespace mlir::vc4;

#include "vc4/Dialect/VC4/VC4OpsDialect.cpp.inc"

void VC4Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "vc4/Dialect/VC4/VC4Ops.cpp.inc"
      >();
}
