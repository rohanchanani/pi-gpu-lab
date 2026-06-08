//===- VC4ValueDialect.cpp - VC4Value dialect definition ------------------===//

#include "vc4/Dialect/VC4Value/IR/VC4ValueDialect.h"

#include "vc4/Dialect/VC4Value/IR/VC4ValueAttrs.h"
#include "vc4/Dialect/VC4Value/IR/VC4ValueOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::vc4value;

#include "vc4/Dialect/VC4Value/IR/VC4ValueDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "vc4/Dialect/VC4Value/IR/VC4ValueAttrDefs.cpp.inc"

void VC4ValueDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "vc4/Dialect/VC4Value/IR/VC4ValueAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "vc4/Dialect/VC4Value/IR/VC4ValueOps.cpp.inc"
      >();
}
