//===- VC4KernelOps.h - VC4Kernel operation declarations -------*- C++ -*-===//

#ifndef VC4_DIALECT_VC4KERNEL_IR_VC4KERNELOPS_H
#define VC4_DIALECT_VC4KERNEL_IR_VC4KERNELOPS_H

#include "vc4/Dialect/VC4Kernel/IR/VC4KernelAttrs.h"
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelDialect.h"
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelOps.h.inc"

#endif // VC4_DIALECT_VC4KERNEL_IR_VC4KERNELOPS_H
