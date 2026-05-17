//===- SSAVC4Ops.h - SSAVC4 operation declarations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VC4_DIALECT_SSAVC4_IR_SSAVC4OPS_H
#define VC4_DIALECT_SSAVC4_IR_SSAVC4OPS_H

#include "vc4/Dialect/SSAVC4/IR/SSAVC4Attrs.h"
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Dialect.h"
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Types.h"
#include "vc4/Dialect/VC4/IR/VC4Attrs.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Ops.h.inc"

#endif // VC4_DIALECT_SSAVC4_IR_SSAVC4OPS_H
