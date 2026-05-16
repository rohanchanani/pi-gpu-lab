//===- SSAVC4Dialect.cpp - SSAVC4 dialect definition ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/SSAVC4/IR/SSAVC4Dialect.h"

#include "vc4/Dialect/SSAVC4/IR/SSAVC4Attrs.h"
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Ops.h"
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Types.h"
#include "vc4/Dialect/VC4/IR/VC4Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ssavc4;

#include "vc4/Dialect/SSAVC4/IR/SSAVC4Dialect.cpp.inc"

// Keep the generated attr/type storage definitions in this translation unit so
// the addAttributes/addTypes template instantiations below see complete
// ImplType definitions.  MLIR's StorageUniquer requires that completeness when
// registering parametric attributes and typedefs.
#define GET_ATTRDEF_CLASSES
#include "vc4/Dialect/SSAVC4/IR/SSAVC4AttrDefs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Types.cpp.inc"

void SSAVC4Dialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "vc4/Dialect/SSAVC4/IR/SSAVC4AttrDefs.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Types.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Ops.cpp.inc"
      >();
}
