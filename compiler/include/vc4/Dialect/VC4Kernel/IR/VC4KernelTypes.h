//===- VC4KernelTypes.h - VC4Kernel dialect types --------------*- C++ -*-===//

#ifndef VC4_DIALECT_VC4KERNEL_IR_VC4KERNELTYPES_H
#define VC4_DIALECT_VC4KERNEL_IR_VC4KERNELTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelTypes.h.inc"

namespace mlir::vc4kernel {

bool isVC4KernelScalarType(Type type);
bool isVC4KernelVector16I32Type(Type type);
bool isVC4KernelVector16F32Type(Type type);
bool isVC4KernelVector16DataType(Type type);
bool isVC4KernelPredType(Type type);
bool isVC4KernelVPMTileType(Type type);
bool isLegalVC4KernelType(Type type);

} // namespace mlir::vc4kernel

#endif // VC4_DIALECT_VC4KERNEL_IR_VC4KERNELTYPES_H
