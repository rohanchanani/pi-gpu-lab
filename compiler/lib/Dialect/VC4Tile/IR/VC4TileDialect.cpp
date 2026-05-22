//===- VC4TileDialect.cpp - VC4Tile dialect definition --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4Tile/IR/VC4TileDialect.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::vc4tile;

#include "vc4/Dialect/VC4Tile/IR/VC4TileDialect.cpp.inc"

void VC4TileDialect::initialize() {}
