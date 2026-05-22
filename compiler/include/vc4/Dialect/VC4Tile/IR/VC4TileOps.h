//===- VC4TileOps.h - VC4Tile operation declarations -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VC4_DIALECT_VC4TILE_IR_VC4TILEOPS_H
#define VC4_DIALECT_VC4TILE_IR_VC4TILEOPS_H

#include "vc4/Dialect/VC4Tile/IR/VC4TileAttrs.h"
#include "vc4/Dialect/VC4Tile/IR/VC4TileDialect.h"
#include "vc4/Dialect/VC4Tile/IR/VC4TileTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4Tile/IR/VC4TileOps.h.inc"

#endif // VC4_DIALECT_VC4TILE_IR_VC4TILEOPS_H
