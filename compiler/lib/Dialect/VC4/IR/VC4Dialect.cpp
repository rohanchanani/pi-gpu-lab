//===- VC4Dialect.cpp - VC4 dialect registration -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4/IR/VC4Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::vc4;

#define GET_ATTRDEF_CLASSES
#include "vc4/Dialect/VC4/IR/VC4AttrDefs.cpp.inc"

#include "vc4/Dialect/VC4/IR/VC4Dialect.cpp.inc"

void VC4Dialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "vc4/Dialect/VC4/IR/VC4AttrDefs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "vc4/Dialect/VC4/IR/VC4Types.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "vc4/Dialect/VC4/IR/VC4Ops.cpp.inc"
      >();
}
