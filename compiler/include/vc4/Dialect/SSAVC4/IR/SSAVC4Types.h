//===- SSAVC4Types.h - SSAVC4 type declarations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VC4_DIALECT_SSAVC4_IR_SSAVC4TYPES_H
#define VC4_DIALECT_SSAVC4_IR_SSAVC4TYPES_H

#include "vc4/Dialect/SSAVC4/IR/SSAVC4Dialect.h"

#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Types.h.inc"

namespace mlir::ssavc4 {

bool isSSAVC4Scalar32(Type type);
bool isSSAVC4Vector16(Type type);
bool isSSAVC4IntCarrier(Type type);
bool isSSAVC4FloatCarrier(Type type);
bool isSSAVC4ValueType(Type type);
bool haveSameSSAVC4Shape(Type lhs, Type rhs);
bool haveSameSSAVC4ElementDomain(Type lhs, Type rhs);

} // namespace mlir::ssavc4

#endif // VC4_DIALECT_SSAVC4_IR_SSAVC4TYPES_H
