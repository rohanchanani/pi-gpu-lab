//===- SSAVC4Dialect.cpp - SSAVC4 dialect implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/SSAVC4/IR/SSAVC4Dialect.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::ssavc4;

#include "vc4/Dialect/SSAVC4/IR/SSAVC4Dialect.cpp.inc"

void SSAVC4Dialect::initialize() {}
