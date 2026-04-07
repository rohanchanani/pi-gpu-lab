#include "vc4/Dialect/VC4/VC4Ops.h"

#include <cassert>

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::vc4;

StringRef mlir::vc4::stringifyBuiltinKind(BuiltinKind kind) {
  switch (kind) {
  case BuiltinKind::qpu_id:
    return "qpu_id";
  case BuiltinKind::num_qpus:
    return "num_qpus";
  }
  llvm_unreachable("unknown VC4 builtin kind");
}

std::optional<BuiltinKind> mlir::vc4::symbolizeBuiltinKind(StringRef kind) {
  return llvm::StringSwitch<std::optional<BuiltinKind>>(kind)
      .Case("qpu_id", BuiltinKind::qpu_id)
      .Case("num_qpus", BuiltinKind::num_qpus)
      .Default(std::nullopt);
}

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4/VC4Ops.cpp.inc"

ParseResult GetBuiltinOp::parse(OpAsmParser &parser, OperationState &result) {
  StringRef keyword;
  Type resultType;
  if (parser.parseKeyword(&keyword) || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(resultType))
    return failure();

  result.addTypes(resultType);
  result.addAttribute("kind", parser.getBuilder().getStringAttr(keyword));
  return success();
}

void GetBuiltinOp::print(OpAsmPrinter &printer) {
  printer << " " << getKindAttr().getValue();
  printer.printOptionalAttrDict((*this)->getAttrs(), {"kind"});
  printer << " : " << getResult().getType();
}

LogicalResult GetBuiltinOp::verify() {
  if (!symbolizeBuiltinKind(getKindAttr().getValue()))
    return emitOpError() << "unsupported builtin kind '" << getKindAttr().getValue()
                         << "', expected one of: qpu_id, num_qpus";
  return success();
}

BuiltinKind GetBuiltinOp::getBuiltinKind() {
  auto kind = symbolizeBuiltinKind(getKindAttr().getValue());
  assert(kind && "GetBuiltinOp verifier must ensure kind is valid");
  return *kind;
}
