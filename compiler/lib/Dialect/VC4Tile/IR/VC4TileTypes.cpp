//===- VC4TileTypes.cpp - VC4Tile dialect types --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4Tile/IR/VC4TileTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::vc4tile;


bool mlir::vc4tile::isVC4TileScalarIdType(Type type) {
  return type && (type.isSignlessInteger(32) || type.isIndex());
}

static bool isVector16Of(Type type, llvm::function_ref<bool(Type)> pred) {
  auto vectorType = llvm::dyn_cast<VectorType>(type);
  if (!vectorType || vectorType.getRank() != 1 ||
      vectorType.getDimSize(0) != 16)
    return false;
  return pred(vectorType.getElementType());
}

bool mlir::vc4tile::isVC4TileVector16I1Type(Type type) {
  return isVector16Of(type, [](Type element) { return element.isInteger(1); });
}

bool mlir::vc4tile::isVC4TileVector16I32Type(Type type) {
  return isVector16Of(type, [](Type element) {
    return element.isSignlessInteger(32);
  });
}

bool mlir::vc4tile::isVC4TileVector16F32Type(Type type) {
  return isVector16Of(type, [](Type element) { return element.isF32(); });
}

bool mlir::vc4tile::isVC4TileVector16DataType(Type type) {
  return isVC4TileVector16I32Type(type) || isVC4TileVector16F32Type(type);
}

bool mlir::vc4tile::isVC4TileSharedTileType(Type type) {
  return type && llvm::isa<SharedTileType>(type);
}

bool mlir::vc4tile::isVC4TileTileType(Type type) {
  return type && llvm::isa<TileType>(type);
}

bool mlir::vc4tile::isVC4TileSupportedM532BitElementType(Type type) {
  if (!type)
    return false;
  if (type.isF32())
    return true;
  if (auto intType = llvm::dyn_cast<IntegerType>(type))
    return intType.getWidth() == 32;
  return false;
}
