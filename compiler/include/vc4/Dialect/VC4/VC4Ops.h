#ifndef VC4_DIALECT_VC4_VC4OPS_H
#define VC4_DIALECT_VC4_VC4OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "vc4/Dialect/VC4/VC4Dialect.h"

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4/VC4Ops.h.inc"

#endif // VC4_DIALECT_VC4_VC4OPS_H
