//===- VC4KernelDialect.cpp - VC4Kernel dialect definition ----------------===//

#include "vc4/Dialect/VC4Kernel/IR/VC4KernelDialect.h"

#include "vc4/Dialect/VC4Kernel/IR/VC4KernelAttrs.h"
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelOps.h"
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelTypes.h"

#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::vc4kernel;

#include "vc4/Dialect/VC4Kernel/IR/VC4KernelDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelAttrDefs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelTypes.cpp.inc"

void VC4KernelDialect::initialize() {
  allowUnknownOperations();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelAttrDefs.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelOps.cpp.inc"
      >();
}
