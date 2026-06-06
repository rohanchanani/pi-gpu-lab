//===- SSAVC4Ops.cpp - SSAVC4 operation implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/SSAVC4/IR/SSAVC4Ops.h"
#include "vc4/Support/VC4ResourceMetadata.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

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

static bool isI32ToF32Reinterpret(Type inputType, Type resultType) {
  return haveSameSSAVC4Shape(inputType, resultType) &&
         ((isSSAVC4IntCarrier(inputType) &&
           isSSAVC4FloatCarrier(resultType)) ||
          (isSSAVC4FloatCarrier(inputType) &&
           isSSAVC4IntCarrier(resultType)));
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

static std::optional<int64_t> getSplatI32Constant(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def || def->getName().getStringRef() != "ssavc4.load_imm" ||
      !value.getType().isSignlessInteger(32))
    return std::nullopt;
  auto attr = dyn_cast_or_null<IntegerAttr>(def->getAttr("value"));
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}



static DictionaryAttr getEnclosingResourceMetadata(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->getName().getStringRef() != "ssavc4.func")
      continue;
    return dyn_cast_or_null<DictionaryAttr>(parent->getAttr("vc4.resource"));
  }
  return nullptr;
}

static LogicalResult verifyVDRResourceMetadata(Operation *op) {
  DictionaryAttr resource = getEnclosingResourceMetadata(op);
  mlir::vc4::SemanticResourceInfo info;
  if (failed(mlir::vc4::parseSemanticResourceMetadata(
          op, resource, info, /*allowAbsent=*/false)))
    return failure();

  if (!info.usesVPM || !info.usesVDR || info.totalVPMRowsPerBlock <= 0 ||
      !info.requiresVPMBaseRowBuiltin)
    return op->emitOpError()
           << "requires semantic vc4.resource VDR/VPM rows and vpm_base_row";

  return success();
}

static LogicalResult verifyOptionalStringAttrChoice(Operation *op,
                                                    StringRef attrName,
                                                    StringRef firstChoice,
                                                    StringRef secondChoice,
                                                    StringRef diagnosticRole) {
  auto attr = dyn_cast_or_null<StringAttr>(op->getAttr(attrName));
  if (!attr)
    return success();
  if (attr.getValue() == firstChoice || attr.getValue() == secondChoice)
    return success();
  return op->emitOpError() << diagnosticRole << " must be one of \""
                           << firstChoice << "\", \"" << secondChoice
                           << "\"; got \"" << attr.getValue() << "\"";
}

static LogicalResult verifyExecutableVPMMode(Operation *op,
                                             VPMElemWidth width,
                                             VPMSubword subword) {
  if (width != VPMElemWidth::w32)
    return op->emitOpError(
        "supports only width = #ssavc4.vpm_elem_width<w32> in executable v1");
  if (subword != VPMSubword::none)
    return op->emitOpError(
        "supports only subword = #ssavc4.vpm_subword<none> in executable v1");
  return success();
}

static LogicalResult verifyExecutableVPMQPUMode(Operation *op,
                                                VPMElemWidth width,
                                                VPMSubword subword) {
  if (width == VPMElemWidth::w32 && subword != VPMSubword::none)
    return op->emitOpError(
        "32-bit VPM QPU access requires subword = #ssavc4.vpm_subword<none>");
  if (width != VPMElemWidth::w32 && subword == VPMSubword::none)
    return op->emitOpError(
        "sub-32 VPM QPU access requires subword = #ssavc4.vpm_subword<packed> or #ssavc4.vpm_subword<laned>");
  return success();
}

static LogicalResult verifyVPMQPUCoordinates(Operation *op,
                                             VPMElemWidth width,
                                             VPMOrientation orientation,
                                             int64_t x, int64_t stride) {
  if (x < 0 || x > 15)
    return op->emitOpError("requires VPM x coordinate in range [0, 15]");
  if (orientation == VPMOrientation::horizontal &&
      width == VPMElemWidth::w32 && x != 0)
    return op->emitOpError("horizontal 32-bit VPM QPU access requires x = 0");
  if (orientation == VPMOrientation::horizontal &&
      width == VPMElemWidth::w16 && x > 1)
    return op->emitOpError(
        "horizontal 16-bit VPM QPU access requires halfword selector x in range [0, 1]");
  if (orientation == VPMOrientation::horizontal &&
      width == VPMElemWidth::w8 && x > 3)
    return op->emitOpError(
        "horizontal 8-bit VPM QPU access requires byte selector x in range [0, 3]");
  if (stride <= 0 || stride > 63)
    return op->emitOpError("requires VPM stride in range [1, 63]");
  return success();
}

static LogicalResult verifyVPMDMACoordinates(Operation *op, int64_t x,
                                             int64_t stride) {
  if (x < 0 || x > 15)
    return op->emitOpError("requires VPM x coordinate in range [0, 15]");
  if (stride <= 0)
    return op->emitOpError("requires positive VPM stride");
  return success();
}

static LogicalResult verifyDynamicRectShape(Operation *op, int64_t maxRows,
                                            int64_t maxCols,
                                            int64_t elemBytes) {
  if (maxRows < 1 || maxRows > 16)
    return op->emitOpError("requires max_rows in range [1, 16]");
  if (maxCols < 1 || maxCols > 16)
    return op->emitOpError("requires max_cols in range [1, 16]");
  if (elemBytes != 4)
    return op->emitOpError("requires elem_bytes = 4");
  return success();
}

static LogicalResult verifyScalarI32Operand(Operation *op, Value value,
                                            StringRef role) {
  if (value.getType().isSignlessInteger(32))
    return success();
  return op->emitOpError() << "requires an i32 " << role << " operand";
}

static LogicalResult verifyDynamicPitchOrStride(Operation *op, Value value,
                                                StringRef role) {
  if (failed(verifyScalarI32Operand(op, value, role)))
    return failure();
  std::optional<int64_t> constant = getSplatI32Constant(value);
  if (!constant)
    return success();
  if (*constant <= 0 || *constant % 4 != 0)
    return op->emitOpError()
           << role << " constant must be positive and 4-byte aligned";
  return success();
}

static LogicalResult verifySuccessorOperands(Operation *op, Block *successor,
                                             OperandRange operands,
                                             StringRef edgeName = "") {
  unsigned operandCount = operands.size();
  unsigned argumentCount = successor->getNumArguments();
  if (operandCount != argumentCount) {
    InFlightDiagnostic diag = op->emitOpError();
    if (!edgeName.empty())
      diag << edgeName << " ";
    diag << "successor operand count does not match target block argument "
            "count";
    return failure();
  }

  for (auto [index, operand] : llvm::enumerate(operands)) {
    Type operandType = operand.getType();
    Type argumentType = successor->getArgument(index).getType();
    if (operandType == argumentType)
      continue;
    InFlightDiagnostic diag = op->emitOpError();
    if (!edgeName.empty())
      diag << edgeName << " ";
    diag << "successor operand type does not match target block argument type"
         << " at index " << index << ": got " << operandType << ", expected "
         << argumentType;
    return failure();
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

LogicalResult UniformReadOp::verify() {
  Operation *op = getOperation();
  if (getIndexAttr().getInt() < 0)
    return emitOpError("'index' must be non-negative");
  return verifyValueType(op, getResult().getType(), "result");
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

LogicalResult VPMWriteOp::verify() {
  Operation *op = getOperation();
  if (!getRow().getType().isSignlessInteger(32))
    return emitOpError("requires an i32 VPM row/y operand");
  if (failed(verifyVector16(op, getValue().getType(), "value")))
    return failure();
  if (failed(verifyExecutableVPMQPUMode(op, getWidth(), getSubword())))
    return failure();
  int64_t lanes = getLanesAttr().getInt();
  if (lanes != 16)
    return emitOpError("supports only full 16-lane VPM vectors in executable v1");
  return verifyVPMQPUCoordinates(op, getWidth(), getOrientation(),
                                 getXAttr().getInt(),
                                 getStrideAttr().getInt());
}

LogicalResult VPMReadOp::verify() {
  Operation *op = getOperation();
  if (!getRow().getType().isSignlessInteger(32))
    return emitOpError("requires an i32 VPM row/y operand");
  if (failed(verifyVector16(op, getResult().getType(), "result")))
    return failure();
  if (failed(verifyExecutableVPMQPUMode(op, getWidth(), getSubword())))
    return failure();
  int64_t lanes = getLanesAttr().getInt();
  if (lanes != 16)
    return emitOpError("supports only full 16-lane VPM vectors in executable v1");
  return verifyVPMQPUCoordinates(op, getWidth(), getOrientation(),
                                 getXAttr().getInt(),
                                 getStrideAttr().getInt());
}

LogicalResult MovOp::verify() {
  Operation *op = getOperation();
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (failed(verifyValueType(op, inputType, "input")) ||
      failed(verifyValueType(op, resultType, "result")))
    return failure();
  if (inputType != resultType)
    if (!isI32ToF32Reinterpret(inputType, resultType))
      return emitOpError()
             << "input and result types must match or be scalar/vector i32/f32 "
                "bit reinterpretation carriers with the same shape; got "
             << inputType << " and " << resultType;
  return success();
}

LogicalResult CondSelectOp::verify() {
  Operation *op = getOperation();
  if (!isa<FlagsType>(getFlags().getType()))
    return emitOpError("flags operand must be !ssavc4.flags");
  if (getCond() == mlir::vc4::Cond::never ||
      getCond() == mlir::vc4::Cond::always) {
    return emitOpError(
        "cond_select requires a real per-lane condition, not never/always");
  }

  Type trueType = getTrueValue().getType();
  Type falseType = getFalseValue().getType();
  Type resultType = getResult().getType();
  if (failed(verifyValueType(op, trueType, "true_value")) ||
      failed(verifyValueType(op, falseType, "false_value")) ||
      failed(verifyValueType(op, resultType, "result")))
    return failure();
  if (trueType != falseType || trueType != resultType) {
    return emitOpError()
           << "true_value, false_value, and result types must match; got "
           << trueType << ", " << falseType << ", and " << resultType;
  }
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
  if (kind == mlir::ssavc4::FlagKind::zero_test && getInputs().size() != 1)
    return emitOpError("zero_test flags require exactly one operand");
  if ((kind == mlir::ssavc4::FlagKind::sub ||
       kind == mlir::ssavc4::FlagKind::compare) &&
      getInputs().size() != 2)
    return emitOpError("sub/compare flags require exactly two operands");
  if (kind == mlir::ssavc4::FlagKind::fsub && getInputs().size() != 2)
    return emitOpError("fsub flags require exactly two operands");

  if (kind == mlir::ssavc4::FlagKind::fsub) {
    if (failed(verifyFloatCarrier(op, firstType, "operand")))
      return failure();
  } else if (failed(verifyIntCarrier(op, firstType, "operand"))) {
    return failure();
  }

  if (getInputs().size() == 2) {
    Type secondType = getInputs()[1].getType();
    if (kind == mlir::ssavc4::FlagKind::fsub) {
      if (failed(verifyFloatCarrier(op, secondType, "operand")) ||
          failed(verifySameShapeAndDomain(op, firstType, secondType, "operand",
                                          "operand")))
        return failure();
    } else if (failed(verifyIntCarrier(op, secondType, "operand")) ||
               failed(verifySameShapeAndDomain(op, firstType, secondType,
                                               "operand", "operand"))) {
      return failure();
    }
  }
  return success();
}

LogicalResult BranchOp::verify() {
  return verifySuccessorOperands(getOperation(), getTarget(),
                                 getTargetOperands());
}

LogicalResult CondBranchOp::verify() {
  Operation *op = getOperation();
  if (!isa<FlagsType>(getFlags().getType()))
    return emitOpError("flags operand must be !ssavc4.flags");
  if (getCond() == mlir::vc4::BranchCond::always)
    return emitOpError("cond_br with always condition is invalid; use ssavc4.br");
  if (failed(verifySuccessorOperands(op, getTrueDest(), getTrueDestOperands(),
                                     "true")))
    return failure();
  return verifySuccessorOperands(op, getFalseDest(), getFalseDestOperands(),
                                 "false");
}

LogicalResult SemaAcquireOp::verify() {
  if (getSemaphore().getType().isSignlessInteger(32))
    return success();
  return emitOpError("requires an i32 semaphore id operand");
}

LogicalResult SemaReleaseOp::verify() {
  if (getSemaphore().getType().isSignlessInteger(32))
    return success();
  return emitOpError("requires an i32 semaphore id operand");
}


LogicalResult VDRLoadOp::verify() {
  Operation *op = getOperation();
  if (!getAddress().getType().isSignlessInteger(32))
    return emitOpError("requires an i32 global base address operand");
  if (!getVpmBaseRow().getType().isSignlessInteger(32))
    return emitOpError("requires an i32 VPM base row operand");

  if (failed(verifyExecutableVPMMode(op, getWidth(), getSubword())))
    return failure();

  int64_t rowLen = getRowLenAttr().getInt();
  if (rowLen <= 0 || rowLen > 16)
    return emitOpError("requires row_len in range [1, 16]");

  int64_t nrows = getNrowsAttr().getInt();
  if (nrows <= 0 || nrows > 16)
    return emitOpError("requires nrows in range [1, 16]");

  int64_t memoryPitchBytes = getMemoryPitchBytesAttr().getInt();
  if (memoryPitchBytes <= 0 || memoryPitchBytes % 4 != 0)
    return emitOpError("requires memory_pitch_bytes to be a positive multiple of 4 bytes");
  if (memoryPitchBytes < rowLen * 4)
    return emitOpError("requires memory_pitch_bytes to cover row_len elements");

  int64_t vpmX = getVpmXAttr().getInt();
  if (vpmX < 0 || vpmX > 15)
    return emitOpError("requires vpm_x in range [0, 15]");

  int64_t vpmPitch = getVpmPitchAttr().getInt();
  if (vpmPitch <= 0 || vpmPitch > 16)
    return emitOpError("requires vpm_pitch in range [1, 16]");

  if (failed(verifyOptionalStringAttrChoice(op, "serialize", "mutex", "none",
                                           "serialize")))
    return failure();
  return verifyVDRResourceMetadata(op);
}

LogicalResult VDRLoadRectDynamicOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyScalarI32Operand(op, getAddress(), "global address")) ||
      failed(verifyScalarI32Operand(op, getVpmBaseRow(), "VPM base row")) ||
      failed(verifyScalarI32Operand(op, getActiveRows(), "active_rows")) ||
      failed(verifyScalarI32Operand(op, getActiveCols(), "active_cols")) ||
      failed(verifyDynamicPitchOrStride(op, getMemoryPitchBytes(),
                                        "memory_pitch_bytes")))
    return failure();

  if (failed(verifyDynamicRectShape(op, getMaxRows(), getMaxCols(),
                                    getElemBytes())))
    return failure();
  if (!getZeroFill())
    return emitOpError("requires zero_fill = true");
  if (failed(verifyExecutableVPMMode(op, getWidth(), getSubword())))
    return failure();
  if (failed(verifyVPMDMACoordinates(op, getDstX(), getVpmPitch())))
    return failure();
  if (getVpmPitch() > 16)
    return emitOpError("requires vpm_pitch in range [1, 16]");
  return verifyOptionalStringAttrChoice(op, "serialize", "mutex", "none",
                                        "serialize");
}

LogicalResult VDWStoreVPMOp::verify() {
  Operation *op = getOperation();
  if (!getAddress().getType().isSignlessInteger(32))
    return emitOpError("requires an i32 global base address operand");
  if (!getVpmY().getType().isSignlessInteger(32))
    return emitOpError("requires an i32 VPM y-coordinate operand");
  if (!getVpmX().getType().isSignlessInteger(32))
    return emitOpError("requires an i32 VPM x-coordinate operand");

  if (failed(verifyExecutableVPMMode(op, getWidth(), getSubword())))
    return failure();
  int64_t rowLen = getRowLenAttr().getInt();
  if (rowLen < 1 || rowLen > 16)
    return emitOpError("requires row_len in range [1, 16]");
  int64_t nrows = getNrowsAttr().getInt();
  if (nrows < 1 || nrows > 16)
    return emitOpError("requires nrows in range [1, 16]");
  int64_t memoryPitchBytes = getMemoryPitchBytesAttr().getInt();
  if (memoryPitchBytes <= 0 || memoryPitchBytes % 4 != 0)
    return emitOpError("requires memory_pitch_bytes to be a positive multiple of 4 bytes");
  if (memoryPitchBytes < rowLen * 4)
    return emitOpError("requires memory_pitch_bytes to cover row_len elements");

  if (getActiveLanesValue() &&
      !getActiveLanesValue().getType().isSignlessInteger(32))
    return emitOpError("requires an i32 dynamic active-lane operand");
  if (getActiveLanesValue() && nrows != 1)
    return emitOpError("supports dynamic active_lanes only for single-row VDW stores");
  if (auto activeLanes = getActiveLanesAttr())
    if (activeLanes.getInt() != rowLen)
      return emitOpError("active_lanes must match row_len for VDW stores");

  if (failed(verifyOptionalStringAttrChoice(op, "serialize", "mutex", "none",
                                           "serialize")))
    return failure();
  return success();
}

LogicalResult VDWStoreRectDynamicOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyScalarI32Operand(op, getAddress(), "global address")) ||
      failed(verifyScalarI32Operand(op, getVpmSourceRow(), "VPM source row")) ||
      failed(verifyScalarI32Operand(op, getActiveRows(), "active_rows")) ||
      failed(verifyScalarI32Operand(op, getActiveCols(), "active_cols")) ||
      failed(verifyDynamicPitchOrStride(op, getMemoryStrideBytes(),
                                        "memory_stride_bytes")))
    return failure();

  if (failed(verifyDynamicRectShape(op, getMaxRows(), getMaxCols(),
                                    getElemBytes())))
    return failure();
  if (!getPreserveInactive())
    return emitOpError("requires preserve_inactive = true");
  if (failed(verifyExecutableVPMMode(op, getWidth(), getSubword())))
    return failure();
  if (failed(verifyVPMDMACoordinates(op, getSrcX(), getVpmPitch())))
    return failure();
  if (getVpmPitch() > 16)
    return emitOpError("requires vpm_pitch in range [1, 16]");
  return verifyOptionalStringAttrChoice(op, "serialize", "mutex", "none",
                                        "serialize");
}

#define GET_OP_CLASSES
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Ops.cpp.inc"
