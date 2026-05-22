//===- VC4TileTypes.cpp - VC4Tile type helpers ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4Tile/IR/VC4TileTypes.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::vc4tile;

static VectorType getRankOneVector16(Type type) {
  auto vectorType = dyn_cast<VectorType>(type);
  if (!vectorType || vectorType.getRank() != 1 || vectorType.getDimSize(0) != 16)
    return VectorType();
  return vectorType;
}

bool mlir::vc4tile::isVC4TileScalarIdType(Type type) {
  return type.isIndex() || type.isSignlessInteger(32);
}

bool mlir::vc4tile::isVC4TileScalar32Type(Type type) {
  return type.isSignlessInteger(32) || type.isF32();
}

bool mlir::vc4tile::isVC4TileVector16I1Type(Type type) {
  VectorType vectorType = getRankOneVector16(type);
  return vectorType && vectorType.getElementType().isSignlessInteger(1);
}

bool mlir::vc4tile::isVC4TileVector16I32Type(Type type) {
  VectorType vectorType = getRankOneVector16(type);
  return vectorType && vectorType.getElementType().isSignlessInteger(32);
}

bool mlir::vc4tile::isVC4TileVector16F32Type(Type type) {
  VectorType vectorType = getRankOneVector16(type);
  return vectorType && vectorType.getElementType().isF32();
}

bool mlir::vc4tile::isVC4TileVector16DataType(Type type) {
  return isVC4TileVector16I32Type(type) || isVC4TileVector16F32Type(type);
}

bool mlir::vc4tile::isVC4TileMaskType(Type type) {
  return isVC4TileVector16I1Type(type);
}

bool mlir::vc4tile::isVC4TileValueType(Type type) {
  return isVC4TileScalarIdType(type) || isVC4TileScalar32Type(type) ||
         isVC4TileVector16DataType(type) || isVC4TileMaskType(type) ||
         isVC4TileSharedTileType(type);
}

bool mlir::vc4tile::isVC4TileSharedTileType(Type type) {
  return isa<SharedTileType>(type);
}

bool mlir::vc4tile::haveSameVC4TileShape(Type lhs, Type rhs) {
  auto lhsVector = dyn_cast<VectorType>(lhs);
  auto rhsVector = dyn_cast<VectorType>(rhs);
  if (!lhsVector && !rhsVector)
    return true;
  if (!lhsVector || !rhsVector)
    return false;
  return lhsVector.getShape() == rhsVector.getShape();
}

static Type getElementDomain(Type type) {
  if (auto vectorType = dyn_cast<VectorType>(type))
    return vectorType.getElementType();
  return type;
}

bool mlir::vc4tile::haveSameVC4TileElementDomain(Type lhs, Type rhs) {
  Type lhsElement = getElementDomain(lhs);
  Type rhsElement = getElementDomain(rhs);
  if (lhsElement.isSignlessInteger(32) && rhsElement.isSignlessInteger(32))
    return true;
  if (lhsElement.isF32() && rhsElement.isF32())
    return true;
  if (lhsElement.isSignlessInteger(1) && rhsElement.isSignlessInteger(1))
    return true;
  return false;
}
