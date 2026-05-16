//===- SSAVC4Ops.cpp - SSAVC4 operation implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/SSAVC4/IR/SSAVC4Ops.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::ssavc4;

namespace {

static LogicalResult emitValueTypeError(Operation *op, Type type,
                                        StringRef role) {
  return op->emitOpError() << role << " type must be one of i32, f32, "
                           << "vector<16xi32>, or vector<16xf32>; got "
                           << type;
}

static LogicalResult verifyValueType(Operation *op, Type type,
                                     StringRef role) {
  if (isSSAVC4ValueType(type))
    return success();
  return emitValueTypeError(op, type, role);
}

static LogicalResult verifyIntCarrier(Operation *op, Type type,
                                      StringRef role) {
  if (isSSAVC4IntCarrier(type))
    return success();
  return op->emitOpError() << role
                           << " type must be an SSAVC4 integer carrier; got "
                           << type;
}

static LogicalResult verifyFloatCarrier(Operation *op, Type type,
                                        StringRef role) {
  if (isSSAVC4FloatCarrier(type))
    return success();
  return op->emitOpError() << role
                           << " type must be an SSAVC4 float carrier; got "
                           << type;
}

static LogicalResult verifySameShape(Operation *op, Type lhs, Type rhs,
                                     StringRef lhsRole, StringRef rhsRole) {
  if (haveSameSSAVC4Shape(lhs, rhs))
    return success();
  return op->emitOpError() << lhsRole << " and " << rhsRole
                           << " must have the same SSAVC4 scalar/vector shape; "
                           << "got " << lhs << " and " << rhs;
}

static LogicalResult verifySameShapeAndDomain(Operation *op, Type lhs, Type rhs,
                                              StringRef lhsRole,
                                              StringRef rhsRole) {
  if (failed(verifySameShape(op, lhs, rhs, lhsRole, rhsRole)))
    return failure();
  if (haveSameSSAVC4ElementDomain(lhs, rhs))
    return success();
  return op->emitOpError() << lhsRole << " and " << rhsRole
                           << " must have the same SSAVC4 element domain; got "
                           << lhs << " and " << rhs;
}

static LogicalResult verifyVector16(Operation *op, Type type, StringRef role) {
  if (isSSAVC4Vector16(type))
    return success();
  return op->emitOpError() << role
                           << " type must be vector<16xi32> or vector<16xf32>; got "
                           << type;
}

static LogicalResult verifyNoUnexpectedPair(Operation *op, Attribute first,
                                            StringRef firstName,
                                            Attribute second,
                                            StringRef secondName) {
  if (first && second) {
    return op->emitOpError() << "must specify only one of '" << firstName
                             << "' or '" << secondName << "'";
  }
  return success();
}

static LogicalResult verifyIntegerAttr32(Operation *op, Attribute attr,
                                         StringRef name) {
  auto intAttr = dyn_cast_or_null<IntegerAttr>(attr);
  if (!intAttr || !intAttr.getType().isSignlessInteger(32)) {
    return op->emitOpError() << "'" << name
                             << "' attribute must be signless i32";
  }
  return success();
}

static bool isUnaryAddOpcode(mlir::vc4::AddOpcode opcode) {
  return opcode == mlir::vc4::AddOpcode::ftoi ||
         opcode == mlir::vc4::AddOpcode::itof ||
         opcode == mlir::vc4::AddOpcode::bit_not ||
         opcode == mlir::vc4::AddOpcode::clz;
}

static bool isFloatBinaryAddOpcode(mlir::vc4::AddOpcode opcode) {
  return opcode == mlir::vc4::AddOpcode::fadd ||
         opcode == mlir::vc4::AddOpcode::fsub ||
         opcode == mlir::vc4::AddOpcode::fmin ||
         opcode == mlir::vc4::AddOpcode::fmax ||
         opcode == mlir::vc4::AddOpcode::fminabs ||
         opcode == mlir::vc4::AddOpcode::fmaxabs;
}

static bool isIntegerBinaryAddOpcode(mlir::vc4::AddOpcode opcode) {
  return opcode == mlir::vc4::AddOpcode::add ||
         opcode == mlir::vc4::AddOpcode::sub ||
         opcode == mlir::vc4::AddOpcode::shr ||
         opcode == mlir::vc4::AddOpcode::asr ||
         opcode == mlir::vc4::AddOpcode::ror ||
         opcode == mlir::vc4::AddOpcode::shl ||
         opcode == mlir::vc4::AddOpcode::min ||
         opcode == mlir::vc4::AddOpcode::max ||
         opcode == mlir::vc4::AddOpcode::bit_and ||
         opcode == mlir::vc4::AddOpcode::bit_or ||
         opcode == mlir::vc4::AddOpcode::bit_xor ||
         opcode == mlir::vc4::AddOpcode::v8adds ||
         opcode == mlir::vc4::AddOpcode::v8subs;
}

} // namespace

LogicalResult LoadImmOp::verify() {
  Operation *op = getOperation();
  Type resultType = getResult().getType();
  if (failed(verifyValueType(op, resultType, "result")))
    return failure();

  Attribute valueAttr = getValueAttr();
  Attribute valuesAttr = getValuesAttr();
  if (failed(verifyNoUnexpectedPair(op, valueAttr, "value", valuesAttr,
                                    "values")))
    return failure();

  switch (getMode()) {
  case mlir::vc4::LoadImmMode::splat32:
    if (!valueAttr)
      return emitOpError("splat32 mode requires a 'value' attribute");
    if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
      if (!floatAttr.getType().isF32())
        return emitOpError("splat32 float 'value' attribute must be f32");
      return success();
    }
    return verifyIntegerAttr32(op, valueAttr, "value");
  case mlir::vc4::LoadImmMode::per_elem_i2:
  case mlir::vc4::LoadImmMode::per_elem_u2: {
    if (!valuesAttr)
      return emitOpError("per-element mode requires a 'values' attribute");
    if (!isa<VectorType>(resultType) || !isSSAVC4IntCarrier(resultType))
      return emitOpError("per-element mode requires result type vector<16xi32>");
    auto dense = dyn_cast<DenseI32ArrayAttr>(valuesAttr);
    if (!dense)
      return emitOpError("per-element mode requires dense i32 array 'values'");
    if (dense.asArrayRef().size() != 16)
      return emitOpError("per-element mode requires exactly 16 lane values");
    int32_t minValue =
        getMode() == mlir::vc4::LoadImmMode::per_elem_i2 ? -2 : 0;
    int32_t maxValue =
        getMode() == mlir::vc4::LoadImmMode::per_elem_i2 ? 1 : 3;
    for (int32_t laneValue : dense.asArrayRef()) {
      if (laneValue < minValue || laneValue > maxValue) {
        return emitOpError() << "lane values for per-element mode must be in "
                             << "range [" << minValue << ", " << maxValue
                             << "]";
      }
    }
    return success();
  }
  }
  llvm_unreachable("unhandled SSAVC4 load immediate mode");
}

LogicalResult ElementNumberOp::verify() {
  Type resultType = getResult().getType();
  if (auto vectorType = dyn_cast<VectorType>(resultType)) {
    if (vectorType.getRank() == 1 && vectorType.getDimSize(0) == 16 &&
        vectorType.getElementType().isSignlessInteger(32))
      return success();
  }
  return emitOpError("result type must be vector<16xi32>");
}

LogicalResult SplatOp::verify() {
  Operation *op = getOperation();
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (!isSSAVC4Scalar32(inputType))
    return emitValueTypeError(op, inputType, "input");
  if (failed(verifyVector16(op, resultType, "result")))
    return failure();
  if (!haveSameSSAVC4ElementDomain(inputType, resultType)) {
    return emitOpError()
           << "input and result must have the same SSAVC4 element domain; got "
           << inputType << " and " << resultType;
  }
  return success();
}

LogicalResult MovOp::verify() {
  Operation *op = getOperation();
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (failed(verifyValueType(op, inputType, "input")) ||
      failed(verifyValueType(op, resultType, "result")))
    return failure();
  if (inputType != resultType)
    return emitOpError() << "input and result types must match; got "
                         << inputType << " and " << resultType;
  return success();
}

LogicalResult ALUAddOp::verify() {
  Operation *op = getOperation();
  Type resultType = getResult().getType();
  if (failed(verifyValueType(op, resultType, "result")))
    return failure();

  unsigned arity = getInputs().size();
  mlir::vc4::AddOpcode opcode = getOpcode();
  if (arity == 0 || arity > 2)
    return emitOpError("requires one or two operands");
  if (isUnaryAddOpcode(opcode) && arity != 1)
    return emitOpError("selected ADD opcode requires exactly one operand");
  if (!isUnaryAddOpcode(opcode) && arity != 2)
    return emitOpError("selected ADD opcode requires exactly two operands");

  Type firstType = getInputs().front().getType();
  if (failed(verifyValueType(op, firstType, "operand")))
    return failure();
  if (opcode == mlir::vc4::AddOpcode::ftoi) {
    if (failed(verifyFloatCarrier(op, firstType, "operand")) ||
        failed(verifyIntCarrier(op, resultType, "result")))
      return failure();
    return verifySameShape(op, firstType, resultType, "operand", "result");
  }
  if (opcode == mlir::vc4::AddOpcode::itof) {
    if (failed(verifyIntCarrier(op, firstType, "operand")) ||
        failed(verifyFloatCarrier(op, resultType, "result")))
      return failure();
    return verifySameShape(op, firstType, resultType, "operand", "result");
  }
  if (opcode == mlir::vc4::AddOpcode::clz ||
      opcode == mlir::vc4::AddOpcode::bit_not) {
    if (failed(verifyIntCarrier(op, firstType, "operand")) ||
        failed(verifyIntCarrier(op, resultType, "result")))
      return failure();
    return verifySameShape(op, firstType, resultType, "operand", "result");
  }

  if (isFloatBinaryAddOpcode(opcode)) {
    if (failed(verifyFloatCarrier(op, firstType, "operand")) ||
        failed(verifyFloatCarrier(op, resultType, "result")))
      return failure();
  } else if (isIntegerBinaryAddOpcode(opcode)) {
    if (failed(verifyIntCarrier(op, firstType, "operand")) ||
        failed(verifyIntCarrier(op, resultType, "result")))
      return failure();
  }

  if (failed(verifySameShapeAndDomain(op, firstType, resultType, "operand",
                                      "result")))
    return failure();
  Type secondType = getInputs()[1].getType();
  if (failed(verifyValueType(op, secondType, "operand")))
    return failure();
  if (isFloatBinaryAddOpcode(opcode)) {
    if (failed(verifyFloatCarrier(op, secondType, "operand")))
      return failure();
  } else if (isIntegerBinaryAddOpcode(opcode)) {
    if (failed(verifyIntCarrier(op, secondType, "operand")))
      return failure();
  }
  if (failed(verifySameShapeAndDomain(op, secondType, resultType, "operand",
                                      "result")))
    return failure();
  return success();
}

LogicalResult ALUMulOp::verify() {
  Operation *op = getOperation();
  Type resultType = getResult().getType();
  if (failed(verifyValueType(op, resultType, "result")))
    return failure();
  if (getInputs().size() != 2)
    return emitOpError("requires exactly two operands");
  for (Value input : getInputs()) {
    if (failed(verifyValueType(op, input.getType(), "operand")) ||
        failed(verifySameShapeAndDomain(op, input.getType(), resultType,
                                        "operand", "result")))
      return failure();
  }
  if (getOpcode() == mlir::vc4::MulOpcode::fmul)
    return verifyFloatCarrier(op, resultType, "result");
  return verifyIntCarrier(op, resultType, "result");
}

LogicalResult PackOp::verify() {
  Operation *op = getOperation();
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (failed(verifyValueType(op, inputType, "input")) ||
      failed(verifyValueType(op, resultType, "result")))
    return failure();
  Attribute mode = getModeAttr();
  if (mode && !isa<mlir::vc4::RegfileAPackModeAttr>(mode) &&
      !isa<mlir::vc4::MulPackModeAttr>(mode)) {
    return emitOpError("'mode' must be a live VC4 pack attribute");
  }
  return verifySameShape(op, inputType, resultType, "input", "result");
}

LogicalResult UnpackOp::verify() {
  Operation *op = getOperation();
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (failed(verifyValueType(op, inputType, "input")) ||
      failed(verifyValueType(op, resultType, "result")))
    return failure();
  Attribute mode = getModeAttr();
  if (mode && !isa<mlir::vc4::RegfileAUnpackModeAttr>(mode) &&
      !isa<mlir::vc4::R4UnpackModeAttr>(mode)) {
    return emitOpError("'mode' must be a live VC4 unpack attribute");
  }
  return verifySameShape(op, inputType, resultType, "input", "result");
}

LogicalResult RotateOp::verify() {
  Operation *op = getOperation();
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (failed(verifyVector16(op, inputType, "input")) ||
      failed(verifyVector16(op, resultType, "result")))
    return failure();
  if (inputType != resultType)
    return emitOpError() << "input and result types must match; got "
                         << inputType << " and " << resultType;
  int64_t amount = getAmountAttr().getInt();
  if (amount < 0 || amount > 15)
    return emitOpError("'amount' must be in range [0, 15]");
  return success();
}

LogicalResult MakeFlagsOp::verify() {
  Operation *op = getOperation();
  if (!getResult().use_empty() && !getResult().hasOneUse())
    return emitOpError("flags values must have at most one use in M3 v1");
  if (getInputs().empty() || getInputs().size() > 2)
    return emitOpError("requires one or two operands");
  mlir::ssavc4::FlagKind kind = getKind();
  Type firstType = getInputs().front().getType();
  if (failed(verifyIntCarrier(op, firstType, "operand")))
    return failure();
  if (kind == mlir::ssavc4::FlagKind::zero_test && getInputs().size() != 1)
    return emitOpError("zero_test flags require exactly one operand");
  if ((kind == mlir::ssavc4::FlagKind::sub ||
       kind == mlir::ssavc4::FlagKind::compare) &&
      getInputs().size() != 2)
    return emitOpError("sub/compare flags require exactly two operands");
  if (getInputs().size() == 2) {
    Type secondType = getInputs()[1].getType();
    if (failed(verifyIntCarrier(op, secondType, "operand")) ||
        failed(verifySameShapeAndDomain(op, firstType, secondType, "operand",
                                        "operand")))
      return failure();
  }
  return success();
}

#define GET_OP_CLASSES
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Ops.cpp.inc"
