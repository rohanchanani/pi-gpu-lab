//===- VC4TileToSSAVC4.h - VC4Tile to SSAVC4 conversion -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VC4_CONVERSION_VC4TILETOSSAVC4_VC4TILETOSSAVC4_H
#define VC4_CONVERSION_VC4TILETOSSAVC4_VC4TILETOSSAVC4_H

#include <memory>

namespace mlir {
class Pass;

namespace vc4 {

std::unique_ptr<Pass> createConvertVC4TileToSSAVC4Pass();
void registerConvertVC4TileToSSAVC4Pass();

} // namespace vc4
} // namespace mlir

#endif // VC4_CONVERSION_VC4TILETOSSAVC4_VC4TILETOSSAVC4_H
