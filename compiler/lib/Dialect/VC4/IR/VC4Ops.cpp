//===- VC4Ops.cpp - VC4 dialect operations -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4/IR/VC4Ops.h"

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

static bool isVC4QPUOp(Operation &op) {
  return op.getName().getStringRef().starts_with("vc4.qpu.");
}

static bool isVC4StructuredOp(Operation &op) {
  StringRef name = op.getName().getStringRef();
  return name.starts_with("vc4.") && !name.starts_with("vc4.qpu.");
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
  auto func = (*this)->getParentOfType<mlir::vc4::FuncOp>();
  if (!func)
    return emitOpError("must be nested in a vc4.func");
  if (!func.getForm() ||
      *func.getForm() != mlir::vc4::FunctionForm::structured)
    return emitOpError("is only legal in functions with form = structured");
  if (!isI32OrVector16I32(getResult().getType()))
    return emitOpError("result type must be i32 or vector<16xi32>");
  return success();
}

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4/IR/VC4Ops.cpp.inc"
