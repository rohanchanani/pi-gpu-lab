//===- SSAVC4Types.cpp - SSAVC4 type helpers -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/SSAVC4/IR/SSAVC4Types.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::ssavc4;

// The generated SSAVC4 typedef method/storage definitions are emitted in
// SSAVC4Dialect.cpp so addTypes sees complete generated storage classes.

static bool isVector16Of(Type type, llvm::function_ref<bool(Type)> pred) {
  auto vectorType = dyn_cast<VectorType>(type);
  return vectorType && vectorType.getRank() == 1 &&
         vectorType.getDimSize(0) == 16 && pred(vectorType.getElementType());
}

bool mlir::ssavc4::isSSAVC4Scalar32(Type type) {
  return type.isSignlessInteger(32) || type.isF32();
}

bool mlir::ssavc4::isSSAVC4Vector16(Type type) {
  return isVector16Of(type, [](Type elementType) {
    return elementType.isSignlessInteger(32) || elementType.isF32();
  });
}

bool mlir::ssavc4::isSSAVC4IntCarrier(Type type) {
  return type.isSignlessInteger(32) ||
         isVector16Of(type, [](Type elementType) {
           return elementType.isSignlessInteger(32);
         });
}

bool mlir::ssavc4::isSSAVC4FloatCarrier(Type type) {
  return type.isF32() || isVector16Of(type, [](Type elementType) {
           return elementType.isF32();
         });
}

bool mlir::ssavc4::isSSAVC4ValueType(Type type) {
  return isSSAVC4Scalar32(type) || isSSAVC4Vector16(type);
}

bool mlir::ssavc4::haveSameSSAVC4Shape(Type lhs, Type rhs) {
  auto lhsVector = dyn_cast<VectorType>(lhs);
  auto rhsVector = dyn_cast<VectorType>(rhs);
  if (static_cast<bool>(lhsVector) != static_cast<bool>(rhsVector))
    return false;
  if (!lhsVector)
    return true;
  return lhsVector.getRank() == 1 && rhsVector.getRank() == 1 &&
         lhsVector.getDimSize(0) == rhsVector.getDimSize(0) &&
         lhsVector.getDimSize(0) == 16;
}

bool mlir::ssavc4::haveSameSSAVC4ElementDomain(Type lhs, Type rhs) {
  return (isSSAVC4IntCarrier(lhs) && isSSAVC4IntCarrier(rhs)) ||
         (isSSAVC4FloatCarrier(lhs) && isSSAVC4FloatCarrier(rhs));
}
