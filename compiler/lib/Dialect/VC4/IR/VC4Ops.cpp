//===- VC4Ops.cpp - VC4 dialect operations -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4/IR/VC4Ops.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

static StringAttr getTypeAttrName(MLIRContext *context) {
  return StringAttr::get(context, "function_type");
}

static StringAttr getArgAttrsName(MLIRContext *context) {
  return StringAttr::get(context, "arg_attrs");
}

static StringAttr getResAttrsName(MLIRContext *context) {
  return StringAttr::get(context, "res_attrs");
}

static bool isI32OrVector16I32(Type type) {
  if (type.isSignlessInteger(32))
    return true;
  auto vectorType = dyn_cast<VectorType>(type);
  if (!vectorType || vectorType.getRank() != 1 || vectorType.isScalable())
    return false;
  return vectorType.getDimSize(0) == 16 &&
         vectorType.getElementType().isSignlessInteger(32);
}

static bool isScalar32BitVC4ValueType(Type type) {
  return type.isSignlessInteger(32) || type.isF32();
}

static bool isVector16Of32BitVC4ValueType(Type type) {
  auto vectorType = dyn_cast<VectorType>(type);
  if (!vectorType || vectorType.getRank() != 1 || vectorType.isScalable())
    return false;
  if (vectorType.getDimSize(0) != 16)
    return false;
  return isScalar32BitVC4ValueType(vectorType.getElementType());
}

static bool isVC4StructuredValueType(Type type) {
  return isScalar32BitVC4ValueType(type) ||
         isVector16Of32BitVC4ValueType(type);
}

static bool hasSameVC4Shape(Type lhs, Type rhs) {
  if (lhs.isSignlessInteger(32) || lhs.isF32())
    return rhs.isSignlessInteger(32) || rhs.isF32();

  auto lhsVector = dyn_cast<VectorType>(lhs);
  auto rhsVector = dyn_cast<VectorType>(rhs);
  if (!lhsVector || !rhsVector)
    return false;
  return lhsVector.getRank() == 1 && rhsVector.getRank() == 1 &&
         !lhsVector.isScalable() && !rhsVector.isScalable() &&
         lhsVector.getDimSize(0) == 16 && rhsVector.getDimSize(0) == 16;
}

static bool isVC4IntValueType(Type type) {
  if (type.isSignlessInteger(32))
    return true;
  auto vectorType = dyn_cast<VectorType>(type);
  return vectorType && vectorType.getRank() == 1 && !vectorType.isScalable() &&
         vectorType.getDimSize(0) == 16 &&
         vectorType.getElementType().isSignlessInteger(32);
}

static bool isVC4FloatValueType(Type type) {
  if (type.isF32())
    return true;
  auto vectorType = dyn_cast<VectorType>(type);
  return vectorType && vectorType.getRank() == 1 && !vectorType.isScalable() &&
         vectorType.getDimSize(0) == 16 && vectorType.getElementType().isF32();
}

static bool isScalarSignlessIntegerOrIndex(Type type) {
  return type.isSignlessIntOrIndex() && !isa<VectorType>(type);
}

static LogicalResult verifyPositiveI32Attr(Operation *op, StringRef attrName,
                                           IntegerAttr attr) {
  if (!attr)
    return success();
  if (!attr.getType().isSignlessInteger(32))
    return op->emitOpError() << "'" << attrName
                             << "' attribute must be signless i32";
  if (attr.getInt() <= 0) {
    return op->emitOpError() << "'" << attrName
                             << "' attribute must be greater than zero";
  }
  return success();
}

static LogicalResult verifyNonNegativeI32Attr(Operation *op, StringRef attrName,
                                              IntegerAttr attr) {
  if (!attr)
    return success();
  if (!attr.getType().isSignlessInteger(32))
    return op->emitOpError() << "'" << attrName
                             << "' attribute must be signless i32";
  if (attr.getInt() < 0) {
    return op->emitOpError() << "'" << attrName
                             << "' attribute must be non-negative";
  }
  return success();
}

static LogicalResult verifyStructuredFormOp(Operation *op) {
  auto func = op->getParentOfType<mlir::vc4::FuncOp>();
  if (!func)
    return op->emitOpError("must be nested in a vc4.func");
  if (!func.getForm() ||
      *func.getForm() != mlir::vc4::FunctionForm::structured)
    return op->emitOpError("is only legal in functions with form = structured");
  return success();
}

static bool isVC4QPUOp(Operation &op) {
  return op.getName().getStringRef().starts_with("vc4.qpu.");
}

static bool isVC4StructuredOp(Operation &op) {
  StringRef name = op.getName().getStringRef();
  return name.starts_with("vc4.") && !name.starts_with("vc4.qpu.");
}

static LogicalResult verifyAllOperandsAndResultAreVC4Values(Operation *op) {
  for (Type operandType : op->getOperandTypes()) {
    if (!isVC4StructuredValueType(operandType)) {
      return op->emitOpError(
          "operands must be i32, f32, vector<16xi32>, or vector<16xf32>");
    }
  }
  for (Type resultType : op->getResultTypes()) {
    if (!isVC4StructuredValueType(resultType)) {
      return op->emitOpError(
          "result type must be i32, f32, vector<16xi32>, or vector<16xf32>");
    }
  }
  return success();
}

static LogicalResult verifySameTypeOperandsAndResult(Operation *op) {
  Type resultType = op->getResult(0).getType();
  for (Type operandType : op->getOperandTypes()) {
    if (operandType != resultType) {
      return op->emitOpError(
                 "requires operand and result types to match exactly")
             << " (got operand type " << operandType << " and result type "
             << resultType << ")";
    }
  }
  return success();
}

static LogicalResult verifyBinaryALUTypes(Operation *op, bool requireFloat) {
  if (op->getNumOperands() != 2)
    return op->emitOpError("expects exactly 2 operands for this opcode");
  if (failed(verifyAllOperandsAndResultAreVC4Values(op)))
    return failure();

  Type lhsType = op->getOperand(0).getType();
  Type rhsType = op->getOperand(1).getType();
  Type resultType = op->getResult(0).getType();
  if (lhsType != rhsType || lhsType != resultType) {
    return op->emitOpError("requires both operands and the result to have the "
                           "same type");
  }

  if (requireFloat) {
    if (!isVC4FloatValueType(resultType))
      return op->emitOpError("requires f32 or vector<16xf32> types");
  } else {
    if (!isVC4IntValueType(resultType))
      return op->emitOpError("requires i32 or vector<16xi32> types");
  }
  return success();
}

static bool haveCompatibleVC4Shapes(TypeRange types) {
  if (types.empty())
    return true;
  Type firstType = types.front();
  for (Type type : types.drop_front()) {
    if (!hasSameVC4Shape(firstType, type))
      return false;
  }
  return true;
}

static std::optional<mlir::vc4::TMUMode> inferTMUModeFromDescriptor(Value value) {
  if (!value || !isa<mlir::vc4::TMUDescType>(value.getType()))
    return std::nullopt;
  if (auto descriptor = value.getDefiningOp<mlir::vc4::TMUDescriptorOp>())
    return descriptor.getMode();
  return std::nullopt;
}

static std::optional<mlir::vc4::VPMDescKind>
inferVPMDescKindFromDescriptor(Value value) {
  if (!value || !isa<mlir::vc4::VPMDescType>(value.getType()))
    return std::nullopt;
  if (auto descriptor = value.getDefiningOp<mlir::vc4::VPMDescriptorOp>())
    return descriptor.getKind();
  return std::nullopt;
}

static std::optional<mlir::vc4::DMADescKind>
inferDMADescKindFromDescriptor(Value value) {
  if (!value || !isa<mlir::vc4::DMADescType>(value.getType()))
    return std::nullopt;
  if (auto descriptor = value.getDefiningOp<mlir::vc4::DMADescriptorOp>())
    return descriptor.getKind();
  return std::nullopt;
}

} // namespace

mlir::vc4::ModuleOp mlir::vc4::ModuleOp::create(Location loc, StringRef name) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  state.addAttribute(::mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  Region *bodyRegion = state.addRegion();
  bodyRegion->push_back(new Block);
  return cast<mlir::vc4::ModuleOp>(Operation::create(state));
}

mlir::vc4::FuncOp mlir::vc4::FuncOp::create(
    Location location, StringRef name, FunctionType type,
    mlir::vc4::ThreadingMode threading, mlir::vc4::FunctionForm form,
    ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  state.addAttribute(::mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(builder.getContext()), TypeAttr::get(type));
  state.addAttribute("threading",
                     mlir::vc4::ThreadingModeAttr::get(builder.getContext(),
                                                       threading));
  state.addAttribute("form",
                     mlir::vc4::FunctionFormAttr::get(builder.getContext(),
                                                      form));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
  return cast<mlir::vc4::FuncOp>(Operation::create(state));
}

ParseResult mlir::vc4::FuncOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> inputs, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag, std::string &) -> Type {
    return FunctionType::get(builder.getContext(), inputs, results);
  };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getTypeAttrName(parser.getContext()), buildFuncType,
      getArgAttrsName(parser.getContext()), getResAttrsName(parser.getContext()));
}

void mlir::vc4::FuncOp::print(OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, *this, /*isVariadic=*/false, "function_type",
      getArgAttrsName(getContext()), getResAttrsName(getContext()));
}

ParseResult mlir::vc4::BuiltinOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  StringRef kindKeyword;
  SMLoc kindLoc = parser.getCurrentLocation();
  if (parser.parseKeyword(&kindKeyword))
    return failure();

  std::optional<mlir::vc4::BuiltinKind> kind =
      mlir::vc4::symbolizeBuiltinKind(kindKeyword);
  if (!kind)
    return parser.emitError(kindLoc)
           << "expected one of [elem_num, qpu_num] for vc4 builtin kind";

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  Type resultType;
  if (parser.parseColonType(resultType))
    return failure();

  result.addTypes(resultType);
  result.addAttribute("kind",
                      mlir::vc4::BuiltinKindAttr::get(parser.getContext(), *kind));
  return success();
}

void mlir::vc4::BuiltinOp::print(OpAsmPrinter &printer) {
  printer << ' ' << mlir::vc4::stringifyBuiltinKind(getKind());
  printer.printOptionalAttrDict((*this)->getAttrs(), {"kind"});
  printer << " : " << getResult().getType();
}

LogicalResult mlir::vc4::ModuleOp::verify() {
  for (Operation &op : getBodyRegion().front()) {
    if (!isa<mlir::vc4::FuncOp>(op))
      return emitOpError("expects only vc4.func operations in the module body");
  }
  return success();
}

LogicalResult mlir::vc4::FuncOp::verify() {
  if (!getThreadingAttr())
    return emitOpError("requires a 'threading' attribute");
  if (!getFormAttr())
    return emitOpError("requires a 'form' attribute");

  mlir::vc4::FunctionForm form = *getForm();
  if (isExternal())
    return success();

  for (Block &block : getBody()) {
    for (Operation &op : block) {
      if (form == mlir::vc4::FunctionForm::structured) {
        if (isVC4QPUOp(op))
          return op.emitOpError(
              "is only legal in functions with form = scheduled");
        if (!isVC4StructuredOp(op))
          return op.emitOpError(
              "is not a legal operation in functions with form = structured");
        continue;
      }

      if (isVC4QPUOp(op))
        continue;

      if (isVC4StructuredOp(op))
        return op.emitOpError(
            "is only legal in functions with form = structured");
      return op.emitOpError(
          "is not a legal operation in functions with form = scheduled");
    }
  }

  return success();
}

LogicalResult mlir::vc4::ReturnOp::verify() {
  auto func = (*this)->getParentOfType<mlir::vc4::FuncOp>();
  if (!func)
    return emitOpError("must be nested in a vc4.func");
  if (!func.getForm() ||
      *func.getForm() != mlir::vc4::FunctionForm::structured)
    return emitOpError("is only legal in functions with form = structured");

  FunctionType functionType = func.getFunctionType();
  if (getNumOperands() != functionType.getNumResults())
    return emitOpError() << "expected " << functionType.getNumResults()
                         << " operands to match the enclosing function signature";

  for (auto [index, operandType, resultType] :
       llvm::zip_equal(llvm::seq<unsigned>(0, getNumOperands()),
                       getOperandTypes(), functionType.getResults())) {
    if (operandType != resultType) {
      return emitOpError() << "type of return operand #" << index << " ("
                           << operandType
                           << ") must match the enclosing function result type ("
                           << resultType << ")";
    }
  }

  return success();
}

LogicalResult mlir::vc4::BuiltinOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();
  if (!isI32OrVector16I32(getResult().getType()))
    return emitOpError("result type must be i32 or vector<16xi32>");
  return success();
}

LogicalResult mlir::vc4::UniformReadOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();
  if (!isVC4StructuredValueType(getResult().getType())) {
    return emitOpError(
        "result type must be i32, f32, vector<16xi32>, or vector<16xf32>");
  }
  return success();
}

LogicalResult mlir::vc4::UniformSeekOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();
  Type offsetType = getOffset().getType();
  if (isa<VectorType>(offsetType))
    return emitOpError("operand must be a scalar signless integer or index");
  if (!offsetType.isSignlessIntOrIndex())
    return emitOpError("operand must be a signless integer or index");
  return success();
}

LogicalResult mlir::vc4::MovOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();
  if (!isVC4StructuredValueType(getResult().getType())) {
    return emitOpError(
        "result type must be i32, f32, vector<16xi32>, or vector<16xf32>");
  }
  return success();
}

LogicalResult mlir::vc4::ReadOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();
  if (!isVC4StructuredValueType(getResult().getType())) {
    return emitOpError(
        "result type must be i32, f32, vector<16xi32>, or vector<16xf32>");
  }
  return success();
}

LogicalResult mlir::vc4::ALUAddOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  switch (getOp()) {
  case mlir::vc4::AddOpcode::nop:
    return emitOpError("structured vc4.alu.add does not support the nop opcode");
  case mlir::vc4::AddOpcode::fadd:
  case mlir::vc4::AddOpcode::fsub:
  case mlir::vc4::AddOpcode::fmin:
  case mlir::vc4::AddOpcode::fmax:
  case mlir::vc4::AddOpcode::fminabs:
  case mlir::vc4::AddOpcode::fmaxabs:
    return verifyBinaryALUTypes(getOperation(), /*requireFloat=*/true);
  case mlir::vc4::AddOpcode::add:
  case mlir::vc4::AddOpcode::sub:
  case mlir::vc4::AddOpcode::shr:
  case mlir::vc4::AddOpcode::asr:
  case mlir::vc4::AddOpcode::ror:
  case mlir::vc4::AddOpcode::shl:
  case mlir::vc4::AddOpcode::min:
  case mlir::vc4::AddOpcode::max:
  case mlir::vc4::AddOpcode::bit_and:
  case mlir::vc4::AddOpcode::bit_or:
  case mlir::vc4::AddOpcode::bit_xor:
  case mlir::vc4::AddOpcode::v8adds:
  case mlir::vc4::AddOpcode::v8subs:
    return verifyBinaryALUTypes(getOperation(), /*requireFloat=*/false);
  case mlir::vc4::AddOpcode::bit_not:
  case mlir::vc4::AddOpcode::clz:
    if (getNumOperands() != 1)
      return emitOpError("expects exactly 1 operand for this opcode");
    if (failed(verifyAllOperandsAndResultAreVC4Values(getOperation())))
      return failure();
    if (failed(verifySameTypeOperandsAndResult(getOperation())))
      return failure();
    if (!isVC4IntValueType(getResult().getType()))
      return emitOpError("requires i32 or vector<16xi32> types");
    return success();
  case mlir::vc4::AddOpcode::ftoi:
    if (getNumOperands() != 1)
      return emitOpError("expects exactly 1 operand for this opcode");
    if (failed(verifyAllOperandsAndResultAreVC4Values(getOperation())))
      return failure();
    if (!isVC4FloatValueType(getOperand(0).getType()))
      return emitOpError("requires an f32 or vector<16xf32> operand");
    if (!isVC4IntValueType(getResult().getType()))
      return emitOpError("requires an i32 or vector<16xi32> result");
    if (!hasSameVC4Shape(getOperand(0).getType(), getResult().getType()))
      return emitOpError("operand and result must have compatible scalar or "
                         "16-lane vector shapes");
    return success();
  case mlir::vc4::AddOpcode::itof:
    if (getNumOperands() != 1)
      return emitOpError("expects exactly 1 operand for this opcode");
    if (failed(verifyAllOperandsAndResultAreVC4Values(getOperation())))
      return failure();
    if (!isVC4IntValueType(getOperand(0).getType()))
      return emitOpError("requires an i32 or vector<16xi32> operand");
    if (!isVC4FloatValueType(getResult().getType()))
      return emitOpError("requires an f32 or vector<16xf32> result");
    if (!hasSameVC4Shape(getOperand(0).getType(), getResult().getType()))
      return emitOpError("operand and result must have compatible scalar or "
                         "16-lane vector shapes");
    return success();
  }

  llvm_unreachable("unhandled vc4.alu.add opcode");
}

LogicalResult mlir::vc4::ALUMulOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  switch (getOp()) {
  case mlir::vc4::MulOpcode::nop:
    return emitOpError("structured vc4.alu.mul does not support the nop opcode");
  case mlir::vc4::MulOpcode::fmul:
    return verifyBinaryALUTypes(getOperation(), /*requireFloat=*/true);
  case mlir::vc4::MulOpcode::mul24:
  case mlir::vc4::MulOpcode::v8muld:
  case mlir::vc4::MulOpcode::v8min:
  case mlir::vc4::MulOpcode::v8max:
  case mlir::vc4::MulOpcode::v8adds:
  case mlir::vc4::MulOpcode::v8subs:
    return verifyBinaryALUTypes(getOperation(), /*requireFloat=*/false);
  }

  llvm_unreachable("unhandled vc4.alu.mul opcode");
}

LogicalResult mlir::vc4::LoadImmOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  Attribute valueAttr = getValueAttr();
  if (!valueAttr)
    return emitOpError("requires a 'value' attribute");

  Type resultType = getResult().getType();
  switch (getMode()) {
  case mlir::vc4::LoadImmMode::splat32: {
    auto intAttr = dyn_cast<IntegerAttr>(valueAttr);
    if (!intAttr || !intAttr.getType().isSignlessInteger(32))
      return emitOpError("splat32 mode requires a signless i32 'value' attribute");
    if (!isVC4StructuredValueType(resultType)) {
      return emitOpError("splat32 mode result type must be i32, f32, "
                         "vector<16xi32>, or vector<16xf32>");
    }
    return success();
  }
  case mlir::vc4::LoadImmMode::per_elem_i2:
  case mlir::vc4::LoadImmMode::per_elem_u2: {
    auto valuesAttr = dyn_cast<DenseI32ArrayAttr>(valueAttr);
    if (!valuesAttr)
      return emitOpError("per-element mode requires a dense i32 array 'value' attribute");
    if (valuesAttr.asArrayRef().size() != 16)
      return emitOpError("per-element mode requires exactly 16 lane values");
    int32_t minValue =
        getMode() == mlir::vc4::LoadImmMode::per_elem_i2 ? -2 : 0;
    int32_t maxValue =
        getMode() == mlir::vc4::LoadImmMode::per_elem_i2 ? 1 : 3;
    for (int32_t laneValue : valuesAttr.asArrayRef()) {
      if (laneValue < minValue || laneValue > maxValue) {
        return emitOpError() << "lane values for mode "
                             << mlir::vc4::stringifyLoadImmMode(getMode())
                             << " must be in range [" << minValue << ", "
                             << maxValue << "]";
      }
    }
    auto vectorType = dyn_cast<VectorType>(resultType);
    if (!vectorType || vectorType.getRank() != 1 || vectorType.isScalable() ||
        vectorType.getDimSize(0) != 16 ||
        !vectorType.getElementType().isSignlessInteger(32)) {
      return emitOpError(
          "per-element mode result type must be vector<16xi32>");
    }
    return success();
  }
  }

  llvm_unreachable("unhandled vc4.load_imm mode");
}

LogicalResult mlir::vc4::PackOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  bool hasRegfileAMode = static_cast<bool>(getRegfileAModeAttr());
  bool hasMulMode = static_cast<bool>(getMulModeAttr());
  if (hasRegfileAMode == hasMulMode) {
    return emitOpError(
        "requires exactly one of 'regfile_a_mode' or 'mul_mode'");
  }

  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (!hasSameVC4Shape(inputType, resultType)) {
    return emitOpError("input and result must have compatible scalar or "
                       "16-lane vector shapes");
  }
  if (!isVC4IntValueType(resultType))
    return emitOpError("result type must be i32 or vector<16xi32>");

  if (hasRegfileAMode) {
    switch (*getRegfileAMode()) {
    case mlir::vc4::RegfileAPackMode::none:
      return emitOpError("regfile_a_mode must not be <none>");
    case mlir::vc4::RegfileAPackMode::to_16a:
    case mlir::vc4::RegfileAPackMode::to_16b:
      if (!isVC4StructuredValueType(inputType)) {
        return emitOpError(
            "input type must be i32, f32, vector<16xi32>, or vector<16xf32>");
      }
      return success();
    case mlir::vc4::RegfileAPackMode::sat32:
    case mlir::vc4::RegfileAPackMode::to_8888:
    case mlir::vc4::RegfileAPackMode::to_8a:
    case mlir::vc4::RegfileAPackMode::to_8b:
    case mlir::vc4::RegfileAPackMode::to_8c:
    case mlir::vc4::RegfileAPackMode::to_8d:
    case mlir::vc4::RegfileAPackMode::sat16a:
    case mlir::vc4::RegfileAPackMode::sat16b:
    case mlir::vc4::RegfileAPackMode::sat8888:
    case mlir::vc4::RegfileAPackMode::sat8a:
    case mlir::vc4::RegfileAPackMode::sat8b:
    case mlir::vc4::RegfileAPackMode::sat8c:
    case mlir::vc4::RegfileAPackMode::sat8d:
      if (!isVC4IntValueType(inputType))
        return emitOpError("regfile_a_mode requires i32 or vector<16xi32> input");
      return success();
    }
    llvm_unreachable("unhandled vc4.pack regfile_a_mode");
  }

  switch (*getMulMode()) {
  case mlir::vc4::MulPackMode::none:
    return emitOpError("mul_mode must not be <none>");
  case mlir::vc4::MulPackMode::to_8888:
  case mlir::vc4::MulPackMode::to_8a:
  case mlir::vc4::MulPackMode::to_8b:
  case mlir::vc4::MulPackMode::to_8c:
  case mlir::vc4::MulPackMode::to_8d:
    if (!isVC4FloatValueType(inputType))
      return emitOpError("mul_mode requires f32 or vector<16xf32> input");
    return success();
  }

  llvm_unreachable("unhandled vc4.pack mul_mode");
}

LogicalResult mlir::vc4::UnpackOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  bool hasRegfileAMode = static_cast<bool>(getRegfileAModeAttr());
  bool hasR4Mode = static_cast<bool>(getR4ModeAttr());
  if (hasRegfileAMode == hasR4Mode) {
    return emitOpError(
        "requires exactly one of 'regfile_a_mode' or 'r4_mode'");
  }

  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (!hasSameVC4Shape(inputType, resultType)) {
    return emitOpError("input and result must have compatible scalar or "
                       "16-lane vector shapes");
  }
  if (!isVC4IntValueType(inputType))
    return emitOpError("input type must be i32 or vector<16xi32>");

  if (hasRegfileAMode) {
    switch (*getRegfileAMode()) {
    case mlir::vc4::RegfileAUnpackMode::none:
      return emitOpError("regfile_a_mode must not be <none>");
    case mlir::vc4::RegfileAUnpackMode::f16a_or_i16a:
    case mlir::vc4::RegfileAUnpackMode::f16b_or_i16b:
    case mlir::vc4::RegfileAUnpackMode::color8a:
    case mlir::vc4::RegfileAUnpackMode::color8b:
    case mlir::vc4::RegfileAUnpackMode::color8c:
    case mlir::vc4::RegfileAUnpackMode::color8d:
      if (!isVC4StructuredValueType(resultType)) {
        return emitOpError(
            "result type must be i32, f32, vector<16xi32>, or vector<16xf32>");
      }
      return success();
    case mlir::vc4::RegfileAUnpackMode::replicate_8d:
      if (!isVC4IntValueType(resultType))
        return emitOpError("replicate_8d requires i32 or vector<16xi32> result");
      return success();
    }
    llvm_unreachable("unhandled vc4.unpack regfile_a_mode");
  }

  switch (*getR4Mode()) {
  case mlir::vc4::R4UnpackMode::none:
    return emitOpError("r4_mode must not be <none>");
  case mlir::vc4::R4UnpackMode::f16a:
  case mlir::vc4::R4UnpackMode::f16b:
  case mlir::vc4::R4UnpackMode::color8a:
  case mlir::vc4::R4UnpackMode::color8b:
  case mlir::vc4::R4UnpackMode::color8c:
  case mlir::vc4::R4UnpackMode::color8d:
    if (!isVC4FloatValueType(resultType))
      return emitOpError("r4_mode requires f32 or vector<16xf32> result");
    return success();
  case mlir::vc4::R4UnpackMode::replicate_8d:
    if (!isVC4IntValueType(resultType))
      return emitOpError("replicate_8d requires i32 or vector<16xi32> result");
    return success();
  }

  llvm_unreachable("unhandled vc4.unpack r4_mode");
}

LogicalResult mlir::vc4::RotateOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (inputType != resultType)
    return emitOpError("input and result types must match exactly");
  if (!isVector16Of32BitVC4ValueType(inputType))
    return emitOpError("input and result type must be vector<16xi32> or vector<16xf32>");

  bool hasAmount = static_cast<bool>(getAmount());
  bool hasImmediate = static_cast<bool>(getImmediateAttr());
  if (hasAmount == hasImmediate) {
    return emitOpError(
        "requires exactly one of an amount operand or an immediate attribute");
  }

  if (hasAmount) {
    if (!isScalarSignlessIntegerOrIndex(getAmount().getType()))
      return emitOpError("amount operand must be a scalar signless integer or index");
    return success();
  }

  auto immediateAttr = getImmediateAttr();
  if (!immediateAttr.getType().isSignlessInteger(32))
    return emitOpError("immediate attribute must be signless i32");
  int64_t value = immediateAttr.getInt();
  if (value < 0 || value > 15)
    return emitOpError("immediate rotate amount must be in range [0, 15]");
  return success();
}

LogicalResult mlir::vc4::TMUDescriptorOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (failed(verifyPositiveI32Attr(getOperation(), "mip_levels",
                                   getMipLevelsAttr())))
    return failure();
  if (failed(
          verifyPositiveI32Attr(getOperation(), "width", getWidthAttr())))
    return failure();
  if (failed(
          verifyPositiveI32Attr(getOperation(), "height", getHeightAttr())))
    return failure();
  if (failed(verifyPositiveI32Attr(getOperation(), "cube_map_stride",
                                   getCubeMapStrideAttr())))
    return failure();

  if (static_cast<bool>(getWidthAttr()) != static_cast<bool>(getHeightAttr())) {
    return emitOpError(
        "requires 'width' and 'height' to be provided together");
  }

  if (getChildImageFieldsAttr() && getChildImageFieldsAttr().empty()) {
    return emitOpError("'child_image_fields' attribute must not be empty");
  }
  if (getBiasFlagsAttr() && getBiasFlagsAttr().empty()) {
    return emitOpError("'bias_flags' attribute must not be empty");
  }

  bool hasTextureOnlyFields = getBaseAttr() || getTextureTypeAttr() ||
                              getMipLevelsAttr() || getWidthAttr() ||
                              getHeightAttr() || getMagFilterAttr() ||
                              getMinFilterAttr() || getWrapSAttr() ||
                              getWrapTAttr() || getFlipYAttr() ||
                              getCubeMapStrideAttr() ||
                              getChildImageFieldsAttr() || getBiasFlagsAttr();

  switch (getMode()) {
  case mlir::vc4::TMUMode::direct:
    if (hasTextureOnlyFields) {
      return emitOpError(
          "direct mode must not carry texture setup attributes");
    }
    return success();
  case mlir::vc4::TMUMode::texture2d:
    if (getCubeMapStrideAttr()) {
      return emitOpError(
          "'cube_map_stride' is only legal for mode = cubemap");
    }
    return success();
  case mlir::vc4::TMUMode::cubemap:
    return success();
  }

  llvm_unreachable("unhandled vc4.tmu.descriptor mode");
}

LogicalResult mlir::vc4::TMURequestOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  ValueRange operands = getOperands();
  if (operands.empty())
    return emitOpError("requires at least one operand");

  unsigned descriptorCount = llvm::count_if(operands, [](Value operand) {
    return isa<mlir::vc4::TMUDescType>(operand.getType());
  });
  if (descriptorCount > 1)
    return emitOpError("accepts at most one !vc4.tmu.desc operand");

  bool hasDescriptor =
      isa<mlir::vc4::TMUDescType>(operands.back().getType());
  if (descriptorCount == 1 && !hasDescriptor) {
    return emitOpError("descriptor operand must be the last operand");
  }

  ValueRange valueOperands =
      hasDescriptor ? operands.drop_back() : operands;
  if (valueOperands.empty()) {
    return emitOpError(
        "requires at least one address or coordinate operand");
  }

  auto verifyTextureOperands = [&](unsigned minCount,
                                   unsigned maxCount) -> LogicalResult {
    if (valueOperands.size() < minCount || valueOperands.size() > maxCount) {
      return emitOpError() << "expects between " << minCount << " and "
                           << maxCount
                           << " coordinate operands for the selected TMU mode";
    }
    TypeRange valueOperandTypes = valueOperands.getTypes();
    for (Type type : valueOperandTypes) {
      if (!isVC4StructuredValueType(type)) {
        return emitOpError("coordinate operands must be i32, f32, "
                           "vector<16xi32>, or vector<16xf32>");
      }
    }
    if (!haveCompatibleVC4Shapes(valueOperandTypes)) {
      return emitOpError("coordinate operands must have compatible scalar or "
                         "16-lane vector shapes");
    }
    return success();
  };

  if (!hasDescriptor) {
    if (valueOperands.size() != 1) {
      return emitOpError(
          "requests without a descriptor are only legal in direct mode and "
          "require exactly one address operand");
    }
    if (!isI32OrVector16I32(valueOperands.front().getType())) {
      return emitOpError(
          "direct-mode address operand must be i32 or vector<16xi32>");
    }
    return success();
  }

  std::optional<mlir::vc4::TMUMode> mode =
      inferTMUModeFromDescriptor(operands.back());
  if (!mode)
    return verifyTextureOperands(/*minCount=*/1, /*maxCount=*/4);

  switch (*mode) {
  case mlir::vc4::TMUMode::direct:
    if (valueOperands.size() != 1) {
      return emitOpError(
          "direct-mode descriptors require exactly one address operand");
    }
    if (!isI32OrVector16I32(valueOperands.front().getType())) {
      return emitOpError(
          "direct-mode address operand must be i32 or vector<16xi32>");
    }
    return success();
  case mlir::vc4::TMUMode::texture2d:
    return verifyTextureOperands(/*minCount=*/1, /*maxCount=*/3);
  case mlir::vc4::TMUMode::cubemap:
    return verifyTextureOperands(/*minCount=*/3, /*maxCount=*/4);
  }

  llvm_unreachable("unhandled vc4.tmu.request mode");
}

LogicalResult mlir::vc4::TMUReadOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (Value token = getToken()) {
    if (Operation *definingOp = token.getDefiningOp()) {
      auto request = dyn_cast<mlir::vc4::TMURequestOp>(definingOp);
      if (!request) {
        return emitOpError(
            "token operand must come from vc4.tmu.request or be a block argument");
      }
      if (request.getUnit() != getUnit()) {
        return emitOpError("token unit must match the selected read unit");
      }
    }
  }

  switch (getPart()) {
  case mlir::vc4::TMUReadPart::raw32:
    if (!isVC4StructuredValueType(getResult().getType())) {
      return emitOpError("part = raw32 requires i32, f32, vector<16xi32>, "
                         "or vector<16xf32> result type");
    }
    return success();
  case mlir::vc4::TMUReadPart::rgba8888:
  case mlir::vc4::TMUReadPart::rg1616:
  case mlir::vc4::TMUReadPart::ba1616:
    if (!isI32OrVector16I32(getResult().getType())) {
      return emitOpError("packed TMU read parts require i32 or vector<16xi32> "
                         "result type");
    }
    return success();
  }

  llvm_unreachable("unhandled vc4.tmu.read part");
}

LogicalResult mlir::vc4::TMUNoSwapOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  bool hasValue = static_cast<bool>(getValue());
  bool hasDisableAttr = static_cast<bool>(getDisableAttr());
  if (hasValue == hasDisableAttr) {
    return emitOpError(
        "requires exactly one of a value operand or a 'disable' attribute");
  }

  if (hasValue && isa<VectorType>(getValue().getType())) {
    return emitOpError("value operand must be a scalar signless integer");
  }

  return success();
}

LogicalResult mlir::vc4::SFUIssueOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (!isVC4StructuredValueType(getInput().getType())) {
    return emitOpError(
        "input type must be i32, f32, vector<16xi32>, or vector<16xf32>");
  }
  return success();
}

LogicalResult mlir::vc4::SFUReadOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (!isVC4StructuredValueType(getResult().getType())) {
    return emitOpError(
        "result type must be i32, f32, vector<16xi32>, or vector<16xf32>");
  }
  return success();
}

LogicalResult mlir::vc4::VPMDescriptorOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (failed(verifyNonNegativeI32Attr(getOperation(), "addr", getAddrAttr())))
    return failure();
  if (failed(
          verifyPositiveI32Attr(getOperation(), "num_vectors",
                                getNumVectorsAttr())))
    return failure();

  switch (getKind()) {
  case mlir::vc4::VPMDescKind::read:
    if (!getNumVectorsAttr())
      return emitOpError("kind = read requires a 'num_vectors' attribute");
    return success();
  case mlir::vc4::VPMDescKind::write:
    if (getNumVectorsAttr())
      return emitOpError("kind = write must not carry 'num_vectors'");
    return success();
  }

  llvm_unreachable("unhandled vc4.vpm.desc kind");
}

LogicalResult mlir::vc4::VPMReadOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (!isVC4StructuredValueType(getResult().getType())) {
    return emitOpError(
        "result type must be i32, f32, vector<16xi32>, or vector<16xf32>");
  }

  std::optional<mlir::vc4::VPMDescKind> kind =
      inferVPMDescKindFromDescriptor(getDescriptor());
  if (kind && *kind != mlir::vc4::VPMDescKind::read)
    return emitOpError("descriptor kind must be <read>");

  return success();
}

LogicalResult mlir::vc4::VPMWriteOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (!isVC4StructuredValueType(getValue().getType())) {
    return emitOpError(
        "value type must be i32, f32, vector<16xi32>, or vector<16xf32>");
  }

  std::optional<mlir::vc4::VPMDescKind> kind =
      inferVPMDescKindFromDescriptor(getDescriptor());
  if (kind && *kind != mlir::vc4::VPMDescKind::write)
    return emitOpError("descriptor kind must be <write>");

  return success();
}

LogicalResult mlir::vc4::DMADescriptorOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (failed(verifyNonNegativeI32Attr(getOperation(), "start_offset",
                                      getStartOffsetAttr())))
    return failure();
  if (failed(verifyNonNegativeI32Attr(getOperation(), "mpitch",
                                      getMpitchAttr())))
    return failure();
  if (failed(verifyNonNegativeI32Attr(getOperation(), "vpitch",
                                      getVpitchAttr())))
    return failure();
  if (failed(
          verifyPositiveI32Attr(getOperation(), "nrows", getNrowsAttr())))
    return failure();
  if (failed(
          verifyPositiveI32Attr(getOperation(), "rowlen", getRowlenAttr())))
    return failure();
  if (failed(
          verifyPositiveI32Attr(getOperation(), "units", getUnitsAttr())))
    return failure();
  if (failed(
          verifyPositiveI32Attr(getOperation(), "depth", getDepthAttr())))
    return failure();
  if (failed(verifyNonNegativeI32Attr(getOperation(), "vpm_base",
                                      getVpmBaseAttr())))
    return failure();
  if (failed(
          verifyNonNegativeI32Attr(getOperation(), "stride", getStrideAttr())))
    return failure();
  if (failed(verifyNonNegativeI32Attr(getOperation(), "extended_stride",
                                      getExtendedStrideAttr())))
    return failure();

  if (auto startOffset = getStartOffsetAttr()) {
    if (startOffset.getInt() > 3) {
      return emitOpError(
          "'start_offset' attribute must be in range [0, 3]");
    }
  }

  if (static_cast<bool>(getMpitchAttr()) != static_cast<bool>(getVpitchAttr())) {
    return emitOpError(
        "requires 'mpitch' and 'vpitch' to be provided together");
  }
  if (static_cast<bool>(getNrowsAttr()) != static_cast<bool>(getRowlenAttr())) {
    return emitOpError(
        "requires 'nrows' and 'rowlen' to be provided together");
  }
  if (getExtendedStrideAttr() && !getStrideAttr()) {
    return emitOpError(
        "'extended_stride' requires the base 'stride' attribute");
  }
  if (getUnitsAttr() && getDepthAttr()) {
    return emitOpError(
        "must not specify both 'units' and 'depth' in one descriptor");
  }

  switch (getKind()) {
  case mlir::vc4::DMADescKind::load:
  case mlir::vc4::DMADescKind::store:
    return success();
  }

  llvm_unreachable("unhandled vc4.dma.desc kind");
}

LogicalResult mlir::vc4::DMAStartOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (!isScalarSignlessIntegerOrIndex(getBase().getType())) {
    return emitOpError("base address must be a scalar signless integer or index");
  }

  std::optional<mlir::vc4::DMADescKind> kind =
      inferDMADescKindFromDescriptor(getDescriptor());
  if (kind && *kind != mlir::vc4::DMADescKind::load &&
      *kind != mlir::vc4::DMADescKind::store) {
    return emitOpError("descriptor kind must be <load> or <store>");
  }
  return success();
}

LogicalResult mlir::vc4::DMAStatusOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  Type resultType = getResult().getType();
  if (!resultType.isSignlessIntOrIndex() || isa<VectorType>(resultType)) {
    return emitOpError("result type must be a scalar signless integer or index");
  }
  return success();
}

LogicalResult mlir::vc4::DMAWaitOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  bool hasToken = static_cast<bool>(getToken());
  bool hasKind = static_cast<bool>(getKindAttr());
  if (hasToken == hasKind) {
    return emitOpError(
        "requires exactly one of a token operand or a 'kind' attribute");
  }

  if (Value token = getToken()) {
    if (Operation *definingOp = token.getDefiningOp()) {
      auto start = dyn_cast<mlir::vc4::DMAStartOp>(definingOp);
      if (!start) {
        return emitOpError(
            "token operand must come from vc4.dma.start or be a block argument");
      }
    }
  }

  return success();
}

LogicalResult mlir::vc4::MutexOp::verify() {
  return verifyStructuredFormOp(getOperation());
}

LogicalResult mlir::vc4::SemaphoreOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  int64_t id = getIdAttr().getInt();
  if (id < 0 || id > 15)
    return emitOpError("semaphore 'id' attribute must be in range [0, 15]");
  return success();
}

LogicalResult mlir::vc4::HostInterruptOp::verify() {
  return verifyStructuredFormOp(getOperation());
}

LogicalResult mlir::vc4::ThreadSwitchOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  auto func = (*this)->getParentOfType<mlir::vc4::FuncOp>();
  if (!func || !func.getThreading() ||
      *func.getThreading() != mlir::vc4::ThreadingMode::threadable) {
    return emitOpError(
        "is only legal in functions with threading = threadable");
  }
  return success();
}

LogicalResult mlir::vc4::ProgramEndOp::verify() {
  return verifyStructuredFormOp(getOperation());
}

LogicalResult mlir::vc4::AsyncWaitOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (getTokens().empty())
    return emitOpError("requires at least one async token operand");
  return success();
}

LogicalResult mlir::vc4::CFBranchOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  unsigned numSuccessors = getNumSuccessors();
  if (getCond() == mlir::vc4::BranchCond::always) {
    if (numSuccessors != 1)
      return emitOpError("cond = always requires exactly one successor");
    return success();
  }

  if (numSuccessors != 2)
    return emitOpError("conditional branch requires exactly two successors");
  return success();
}

mlir::SuccessorOperands mlir::vc4::CFBranchOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "successor index out of range");
  return mlir::SuccessorOperands(
      mlir::MutableOperandRange(getOperation(), /*start=*/0, /*length=*/0));
}

LogicalResult mlir::vc4::EnqueueQPUOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (getNumOperands() != 0 && getNumOperands() != 2) {
    return emitOpError(
        "supports either no operands or exactly two operands for uniforms base and length");
  }
  if (getNumOperands() == 2) {
    if (!isScalarSignlessIntegerOrIndex(getOperand(0).getType()) ||
        !isScalarSignlessIntegerOrIndex(getOperand(1).getType())) {
      return emitOpError(
          "uniforms base and length operands must be scalar signless integers or index");
    }
  }

  Operation *symbol = SymbolTable::lookupNearestSymbolFrom(getOperation(), getEntryAttr());
  auto func = dyn_cast_or_null<mlir::vc4::FuncOp>(symbol);
  if (!func)
    return emitOpError("referenced 'entry' must resolve to a vc4.func symbol");
  if (!func.getKernelAttr())
    return emitOpError("referenced function must be marked with the 'kernel' attribute");
  return success();
}

LogicalResult mlir::vc4::ReserveQPUOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  int64_t mask = getMaskAttr().getInt();
  if (mask < 0 || mask > 0xFFF)
    return emitOpError("mask attribute must fit the 12-QPU target range [0, 4095]");
  return success();
}

LogicalResult mlir::vc4::V3DQueryOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  auto verifyScalarIntLike = [&](Type type, StringRef what) -> LogicalResult {
    if (!isScalarSignlessIntegerOrIndex(type)) {
      return emitOpError() << what
                           << " must be a scalar signless integer or index";
    }
    return success();
  };

  if (getNumResults() != 1)
    return emitOpError("currently requires exactly one result");
  if (failed(verifyScalarIntLike(getResult(0).getType(), "result type")))
    return failure();

  switch (getKind()) {
  case mlir::vc4::V3DQueryKind::ident:
  case mlir::vc4::V3DQueryKind::queue_status:
  case mlir::vc4::V3DQueryKind::interrupt_status:
  case mlir::vc4::V3DQueryKind::error_status:
    if (getNumOperands() != 0)
      return emitOpError("selected query kind does not accept selector operands");
    return success();
  case mlir::vc4::V3DQueryKind::perf_counter:
  case mlir::vc4::V3DQueryKind::scratch:
    if (getNumOperands() != 1)
      return emitOpError("selected query kind requires exactly one selector operand");
    return verifyScalarIntLike(getOperand(0).getType(), "selector operand");
  }

  llvm_unreachable("unhandled vc4.v3d.query kind");
}

LogicalResult mlir::vc4::V3DConfigureOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  auto verifyScalarOperands = [&](unsigned expected) -> LogicalResult {
    if (getNumOperands() != expected) {
      return emitOpError() << "selected configure kind requires exactly "
                           << expected << " payload operand"
                           << (expected == 1 ? "" : "s");
    }
    for (Type type : getOperandTypes()) {
      if (!isScalarSignlessIntegerOrIndex(type)) {
        return emitOpError(
            "payload operands must be scalar signless integers or index");
      }
    }
    return success();
  };

  switch (getKind()) {
  case mlir::vc4::V3DConfigureKind::cache_control:
  case mlir::vc4::V3DConfigureKind::interrupt_enable:
  case mlir::vc4::V3DConfigureKind::interrupt_disable:
  case mlir::vc4::V3DConfigureKind::perf_enable:
  case mlir::vc4::V3DConfigureKind::vpm_reservation:
  case mlir::vc4::V3DConfigureKind::vpm_allocator:
    return verifyScalarOperands(/*expected=*/1);
  case mlir::vc4::V3DConfigureKind::perf_map:
  case mlir::vc4::V3DConfigureKind::scratch:
    return verifyScalarOperands(/*expected=*/2);
  case mlir::vc4::V3DConfigureKind::perf_clear:
    return verifyScalarOperands(/*expected=*/0);
  }

  llvm_unreachable("unhandled vc4.v3d.configure kind");
}

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4/IR/VC4Ops.cpp.inc"
