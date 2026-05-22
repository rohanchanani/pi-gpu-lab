//===- VC4TileDialect.cpp - VC4Tile dialect definition --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4Tile/IR/VC4TileDialect.h"

#include "vc4/Dialect/VC4Tile/IR/VC4TileAttrs.h"
#include "vc4/Dialect/VC4Tile/IR/VC4TileOps.h"
#include "vc4/Dialect/VC4Tile/IR/VC4TileTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::vc4tile;

#include "vc4/Dialect/VC4Tile/IR/VC4TileDialect.cpp.inc"

// Keep generated attribute and type storage definitions in this translation
// unit. MLIR instantiates StorageUniquer registration from addAttributes and
// addTypes in initialize(); that registration requires each generated ImplType
// to be complete at the call site rather than only forward-declared from the
// public .h.inc files.
#define GET_ATTRDEF_CLASSES
#include "vc4/Dialect/VC4Tile/IR/VC4TileAttrDefs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "vc4/Dialect/VC4Tile/IR/VC4TileTypes.cpp.inc"

void VC4TileDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "vc4/Dialect/VC4Tile/IR/VC4TileAttrDefs.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "vc4/Dialect/VC4Tile/IR/VC4TileTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "vc4/Dialect/VC4Tile/IR/VC4TileOps.cpp.inc"
      >();
}
