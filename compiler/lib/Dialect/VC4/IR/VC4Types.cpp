//===- VC4Types.cpp - VC4 dialect types ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4/IR/VC4Dialect.h"
#include "vc4/Dialect/VC4/IR/VC4Types.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::vc4;

#define GET_TYPEDEF_CLASSES
#include "vc4/Dialect/VC4/IR/VC4Types.cpp.inc"
