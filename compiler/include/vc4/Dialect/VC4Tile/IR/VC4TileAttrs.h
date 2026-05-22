//===- VC4TileAttrs.h - VC4Tile attributes ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VC4_DIALECT_VC4TILE_IR_VC4TILEATTRS_H
#define VC4_DIALECT_VC4TILE_IR_VC4TILEATTRS_H

#include "vc4/Dialect/VC4Tile/IR/VC4TileDialect.h"

#include "mlir/IR/Attributes.h"

#include "vc4/Dialect/VC4Tile/IR/VC4TileEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "vc4/Dialect/VC4Tile/IR/VC4TileAttrDefs.h.inc"

#endif // VC4_DIALECT_VC4TILE_IR_VC4TILEATTRS_H
