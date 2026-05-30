//===- VC4KernelAttrs.h - VC4Kernel attributes -----------------*- C++ -*-===//

#ifndef VC4_DIALECT_VC4KERNEL_IR_VC4KERNELATTRS_H
#define VC4_DIALECT_VC4KERNEL_IR_VC4KERNELATTRS_H

#include "vc4/Dialect/VC4Kernel/IR/VC4KernelDialect.h"

#include "mlir/IR/Attributes.h"

#include "vc4/Dialect/VC4Kernel/IR/VC4KernelEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelAttrDefs.h.inc"

#endif // VC4_DIALECT_VC4KERNEL_IR_VC4KERNELATTRS_H
