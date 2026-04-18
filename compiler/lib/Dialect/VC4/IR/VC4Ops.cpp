//===- VC4Ops.cpp - VC4 dialect operations -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4/IR/VC4Ops.h"

#include "mlir/IR/BuiltinAttributes.h"
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
  if (!offsetType.isSignlessIntOrIndex())
    return emitOpError("operand must be a signless integer or index");
  if (isa<VectorType>(offsetType))
    return emitOpError("operand must be a scalar signless integer or index");
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

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4/IR/VC4Ops.cpp.inc"
