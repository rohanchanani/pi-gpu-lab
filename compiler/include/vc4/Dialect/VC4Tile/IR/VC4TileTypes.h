//===- VC4TileTypes.h - VC4Tile types --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VC4_DIALECT_VC4TILE_IR_VC4TILETYPES_H
#define VC4_DIALECT_VC4TILE_IR_VC4TILETYPES_H

#include "vc4/Dialect/VC4Tile/IR/VC4TileDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "vc4/Dialect/VC4Tile/IR/VC4TileTypes.h.inc"

namespace mlir::vc4tile {

bool isVC4TileScalarIdType(Type type);
bool isVC4TileScalar32Type(Type type);
bool isVC4TileVector16I1Type(Type type);
bool isVC4TileVector16I32Type(Type type);
bool isVC4TileVector16F32Type(Type type);
bool isVC4TileVector16DataType(Type type);
bool isVC4TileMaskType(Type type);
bool isVC4TileValueType(Type type);
bool isVC4TileSharedTileType(Type type);
bool haveSameVC4TileShape(Type lhs, Type rhs);
bool haveSameVC4TileElementDomain(Type lhs, Type rhs);

} // namespace mlir::vc4tile

#endif // VC4_DIALECT_VC4TILE_IR_VC4TILETYPES_H
