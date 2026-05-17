//===- SSAVC4ToVC4.h - SSAVC4 to scheduled VC4 conversion ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VC4_CONVERSION_SSAVC4TOVC4_SSAVC4TOVC4_H
#define VC4_CONVERSION_SSAVC4TOVC4_SSAVC4TOVC4_H

#include <memory>

namespace mlir {
class Pass;

namespace vc4 {

std::unique_ptr<Pass> createConvertSSAVC4ToVC4Pass();
void registerConvertSSAVC4ToVC4Pass();

} // namespace vc4
} // namespace mlir

#endif // VC4_CONVERSION_SSAVC4TOVC4_SSAVC4TOVC4_H
