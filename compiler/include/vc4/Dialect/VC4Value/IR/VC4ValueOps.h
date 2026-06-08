//===- VC4ValueOps.h - VC4Value operation declarations ----------*- C++ -*-===//

#ifndef VC4_DIALECT_VC4VALUE_IR_VC4VALUEOPS_H
#define VC4_DIALECT_VC4VALUE_IR_VC4VALUEOPS_H

#include "vc4/Dialect/VC4Value/IR/VC4ValueDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4Value/IR/VC4ValueOps.h.inc"

#endif // VC4_DIALECT_VC4VALUE_IR_VC4VALUEOPS_H
