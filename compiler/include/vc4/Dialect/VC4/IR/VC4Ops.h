//===- VC4Ops.h - VC4 dialect operations -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VC4_DIALECT_VC4_IR_VC4OPS_H
#define VC4_DIALECT_VC4_IR_VC4OPS_H

#include "vc4/Dialect/VC4/IR/VC4Dialect.h"
#include "vc4/Dialect/VC4/IR/VC4Enums.h"
#include "vc4/Dialect/VC4/IR/VC4Types.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4/IR/VC4Ops.h.inc"

#endif // VC4_DIALECT_VC4_IR_VC4OPS_H
