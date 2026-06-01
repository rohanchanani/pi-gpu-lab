//===- VC4KernelOps.cpp - VC4Kernel operation implementation --------------===//

#include "vc4/Dialect/VC4Kernel/IR/VC4KernelOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSet.h"

#include <optional>

using namespace mlir;
using namespace mlir::vc4kernel;

namespace {

static LogicalResult emitTypeError(Operation *op, Type type, StringRef role,
                                   StringRef expected) {
  return op->emitOpError() << role << " type must be " << expected << "; got "
                           << type;
}

static bool hasName(Operation *op, StringRef name) {
  return op && op->getName().getStringRef() == name;
}

static bool isParentKernel(Operation *op) {
  return op->getParentOfType<KernelOp>() != nullptr;
}

static LogicalResult verifyNoUniformIndex(Operation *op, StringRef name) {
  if (!op->getAttr("uniform_index"))
    return success();
  return op->emitOpError()
         << name << " must not carry uniform_index; launch ABI slots are "
         << "assigned only during vc4kernel -> ssavc4 lowering";
}

static LogicalResult verifyPred16(Operation *op, Type type, StringRef role) {
  if (isVC4KernelPredType(type))
    return success();
  return emitTypeError(op, type, role, "!vc4kernel.pred<16>");
}

static LogicalResult verifyVector16Data(Operation *op, Type type,
                                        StringRef role) {
  if (isVC4KernelVector16DataType(type))
    return success();
  return emitTypeError(op, type, role,
                       "vector<16xi32> or vector<16xf32>");
}

static LogicalResult verifySameType(Operation *op, Type lhs, Type rhs,
                                    StringRef what) {
  if (lhs == rhs)
    return success();
  return op->emitOpError() << what << " must have matching types; got " << lhs
                           << " and " << rhs;
}

static StringAttr asStringAttr(Attribute attr) {
  return llvm::dyn_cast_if_present<StringAttr>(attr);
}

static ArrayAttr asArrayAttr(Attribute attr) {
  return llvm::dyn_cast_if_present<ArrayAttr>(attr);
}

static DictionaryAttr asDictionaryAttr(Attribute attr) {
  return llvm::dyn_cast_if_present<DictionaryAttr>(attr);
}

static IntegerAttr asIntegerAttr(Attribute attr) {
  return llvm::dyn_cast_if_present<IntegerAttr>(attr);
}

static StringAttr getStringAttr(DictionaryAttr dict, StringRef name) {
  return dict ? asStringAttr(dict.get(name)) : nullptr;
}

static StringAttr getKernelSymNameAttr(KernelOp kernel) {
  return asStringAttr(kernel->getAttr(SymbolTable::getSymbolAttrName()));
}

static StringAttr getKernelPublicNameAttr(KernelOp kernel) {
  return asStringAttr(kernel->getAttr("public_name"));
}

static ScheduleModeAttr getKernelScheduleModeAttr(KernelOp kernel) {
  return llvm::dyn_cast_if_present<ScheduleModeAttr>(
      kernel->getAttr("schedule_mode"));
}

static ArrayAttr getKernelArgAttrsAttr(KernelOp kernel) {
  return asArrayAttr(kernel->getAttr("arg_attrs"));
}

static IntegerAttr getKernelWarpsPerBlockAttr(KernelOp kernel) {
  return asIntegerAttr(kernel->getAttr("warps_per_block"));
}

static std::optional<int64_t> getConstantI32(Value value) {
  Operation *def = value.getDefiningOp();
  if (!hasName(def, "arith.constant"))
    return std::nullopt;
  auto attr = def->getAttrOfType<IntegerAttr>("value");
  if (!attr || !value.getType().isSignlessInteger(32))
    return std::nullopt;
  return attr.getInt();
}

static bool isKnownScalarByteOffsetAligned4Impl(Value value,
                                                unsigned depth);
static bool isKnownVectorByteOffsetsAligned4Impl(Value value,
                                                 unsigned depth);

static bool isKnownScalarByteOffsetAligned4(Value value) {
  return isKnownScalarByteOffsetAligned4Impl(value, 0);
}

static bool isKnownVectorByteOffsetsAligned4(Value value) {
  return isKnownVectorByteOffsetsAligned4Impl(value, 0);
}

static bool isKnownScalarByteOffsetAligned4Impl(Value value, unsigned depth) {
  if (depth > 16)
    return false;
  std::optional<int64_t> constant = getConstantI32(value);
  if (constant)
    return *constant % 4 == 0;

  Operation *def = value.getDefiningOp();
  if (!def || !value.getType().isSignlessInteger(32))
    return false;

  if (hasName(def, "arith.addi") || hasName(def, "arith.subi"))
    return def->getNumOperands() == 2 &&
           isKnownScalarByteOffsetAligned4Impl(def->getOperand(0), depth + 1) &&
           isKnownScalarByteOffsetAligned4Impl(def->getOperand(1), depth + 1);

  if (hasName(def, "arith.shli") && def->getNumOperands() == 2) {
    std::optional<int64_t> amount = getConstantI32(def->getOperand(1));
    return amount && *amount >= 2;
  }

  if (hasName(def, "arith.muli") && def->getNumOperands() == 2) {
    std::optional<int64_t> lhs = getConstantI32(def->getOperand(0));
    std::optional<int64_t> rhs = getConstantI32(def->getOperand(1));
    return (lhs && *lhs % 4 == 0) || (rhs && *rhs % 4 == 0);
  }

  return false;
}

static bool isLaneBytes(Value value) {
  Operation *def = value.getDefiningOp();
  if (!hasName(def, "vc4kernel.fragment_shl") || def->getNumOperands() != 2)
    return false;
  Operation *laneRange = def->getOperand(0).getDefiningOp();
  if (!hasName(laneRange, "vc4kernel.lane_range"))
    return false;
  std::optional<int64_t> amount = getConstantI32(def->getOperand(1));
  return amount && *amount == 2;
}

static bool isLaneBytesOrGreater(Value value) {
  Operation *def = value.getDefiningOp();
  if (!hasName(def, "vc4kernel.fragment_shl") || def->getNumOperands() != 2)
    return false;
  Operation *laneRange = def->getOperand(0).getDefiningOp();
  if (!hasName(laneRange, "vc4kernel.lane_range"))
    return false;
  std::optional<int64_t> amount = getConstantI32(def->getOperand(1));
  return amount && *amount >= 2;
}

static bool isKnownVectorByteOffsetsAligned4Impl(Value value, unsigned depth) {
  if (depth > 16)
    return false;
  if (isLaneBytesOrGreater(value))
    return true;

  Operation *def = value.getDefiningOp();
  if (!def)
    return false;

  if (hasName(def, "vc4kernel.splat") && def->getNumOperands() == 1)
    return isKnownScalarByteOffsetAligned4Impl(def->getOperand(0), depth + 1);

  if ((hasName(def, "vc4kernel.fragment_add") ||
       hasName(def, "vc4kernel.fragment_sub")) &&
      def->getNumOperands() == 2)
    return isKnownVectorByteOffsetsAligned4Impl(def->getOperand(0), depth + 1) &&
           isKnownVectorByteOffsetsAligned4Impl(def->getOperand(1), depth + 1);

  return false;
}

static bool isContiguousByteOffsets(Value value) {
  if (isLaneBytes(value))
    return true;
  Operation *def = value.getDefiningOp();
  if (!hasName(def, "vc4kernel.fragment_add") || def->getNumOperands() != 2)
    return false;
  auto isBasePlusLaneBytes = [](Value lhs, Value rhs) {
    Operation *splat = lhs.getDefiningOp();
    return hasName(splat, "vc4kernel.splat") && isLaneBytes(rhs);
  };
  return isBasePlusLaneBytes(def->getOperand(0), def->getOperand(1)) ||
         isBasePlusLaneBytes(def->getOperand(1), def->getOperand(0));
}

enum class PredNormalForm { Full, Empty, Tail, RectRow };

static std::optional<PredNormalForm> normalizePredExpression(Value pred) {
  Operation *def = pred.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (hasName(def, "vc4kernel.pred.full"))
    return PredNormalForm::Full;
  if (hasName(def, "vc4kernel.pred.empty"))
    return PredNormalForm::Empty;
  if (hasName(def, "vc4kernel.pred.tail"))
    return PredNormalForm::Tail;
  if (hasName(def, "vc4kernel.pred.rect"))
    return PredNormalForm::RectRow;
  if (hasName(def, "vc4kernel.pred.not")) {
    if (def->getNumOperands() != 1)
      return std::nullopt;
    std::optional<PredNormalForm> input =
        normalizePredExpression(def->getOperand(0));
    if (!input)
      return std::nullopt;
    if (*input == PredNormalForm::Full)
      return PredNormalForm::Empty;
    if (*input == PredNormalForm::Empty)
      return PredNormalForm::Full;
    return std::nullopt;
  }
  if (hasName(def, "vc4kernel.pred.and") ||
      hasName(def, "vc4kernel.pred.or")) {
    if (def->getNumOperands() != 2)
      return std::nullopt;
    Value lhs = def->getOperand(0);
    Value rhs = def->getOperand(1);
    std::optional<PredNormalForm> lhsForm = normalizePredExpression(lhs);
    std::optional<PredNormalForm> rhsForm = normalizePredExpression(rhs);
    if (!lhsForm || !rhsForm)
      return std::nullopt;
    if (hasName(def, "vc4kernel.pred.and")) {
      if (*lhsForm == PredNormalForm::Empty ||
          *rhsForm == PredNormalForm::Empty)
        return PredNormalForm::Empty;
      if (*lhsForm == PredNormalForm::Full)
        return rhsForm;
      if (*rhsForm == PredNormalForm::Full)
        return lhsForm;
      if (lhs == rhs)
        return lhsForm;
      return std::nullopt;
    }
    if (*lhsForm == PredNormalForm::Full ||
        *rhsForm == PredNormalForm::Full)
      return PredNormalForm::Full;
    if (*lhsForm == PredNormalForm::Empty)
      return rhsForm;
    if (*rhsForm == PredNormalForm::Empty)
      return lhsForm;
    if (lhs == rhs)
      return lhsForm;
    return std::nullopt;
  }
  return std::nullopt;
}

static bool isNormalizablePredExpression(Value pred) {
  return normalizePredExpression(pred).has_value();
}

static std::optional<int64_t> getVPMAllocRows(Value tile) {
  Operation *def = tile.getDefiningOp();
  if (!hasName(def, "vc4kernel.vpm_alloc"))
    return std::nullopt;
  auto rows = def->getAttrOfType<IntegerAttr>("rows");
  if (!rows)
    return std::nullopt;
  return rows.getInt();
}

static LogicalResult verifyVPMRowInBounds(Operation *op, Value tile, Value row,
                                          int64_t span = 1) {
  Operation *alloc = tile.getDefiningOp();
  if (!hasName(alloc, "vc4kernel.vpm_alloc"))
    return op->emitOpError(
        "VPM tile operand must be produced by vc4kernel.vpm_alloc");

  std::optional<int64_t> rows = getVPMAllocRows(tile);
  if (!rows)
    return op->emitOpError(
        "VPM tile operand must be produced by vc4kernel.vpm_alloc");
  std::optional<int64_t> rowCst = getConstantI32(row);
  if (!rowCst)
    return op->emitOpError("VPM row must be a scalar i32 constant in Stage 1");
  if (*rowCst < 0 || *rowCst + span > *rows)
    return op->emitOpError("VPM row access is out of bounds for allocation");
  return success();
}

static LogicalResult verifyOrientationRowOnly(Operation *op) {
  auto orientation =
      llvm::dyn_cast_if_present<VPMOrientationAttr>(op->getAttr("orientation"));
  if (!orientation)
    return op->emitOpError("orientation attribute is required");
  if (orientation.getValue() == VPMOrientation::row)
    return success();
  return op->emitOpError(
      "column VPM orientation is not supported by Stage 1 lowering");
}

static LogicalResult verifyArgAttrs(KernelOp kernel, Block &entry) {
  ArrayAttr argAttrs = getKernelArgAttrsAttr(kernel);
  if (!argAttrs)
    return kernel.emitOpError("arg_attrs must be an array of dictionaries");
  if (argAttrs.size() != entry.getNumArguments())
    return kernel.emitOpError("arg_attrs length must match formal argument count");

  llvm::StringSet<> names;
  for (auto indexed : llvm::enumerate(argAttrs)) {
    auto dict = asDictionaryAttr(indexed.value());
    if (!dict)
      return kernel.emitOpError("arg_attrs entries must be dictionaries");
    if (dict.get("uniform_index"))
      return kernel.emitOpError("arg_attrs must not contain uniform_index");

    auto name = getStringAttr(dict, "name");
    auto kind = getStringAttr(dict, "kind");
    auto direction = getStringAttr(dict, "direction");
    if (!name || name.getValue().empty())
      return kernel.emitOpError("arg_attrs entries require non-empty name");
    if (!names.insert(name.getValue()).second)
      return kernel.emitOpError() << "duplicate arg_attrs name '"
                                  << name.getValue() << "'";
    if (!kind || (kind.getValue() != "scalar" && kind.getValue() != "buffer"))
      return kernel.emitOpError(
          "arg_attrs kind must be 'scalar' or 'buffer'");
    if (!direction ||
        (direction.getValue() != "by_value" && direction.getValue() != "in" &&
         direction.getValue() != "out" && direction.getValue() != "inout"))
      return kernel.emitOpError(
          "arg_attrs direction must be by_value, in, out, or inout");

    Type formalType = entry.getArgument(indexed.index()).getType();
    if (kind.getValue() == "buffer") {
      if (!formalType.isSignlessInteger(32))
        return kernel.emitOpError("buffer args must be i32 raw device pointers");
      if (!getStringAttr(dict, "elem_type") || getStringAttr(dict, "type"))
        return kernel.emitOpError(
            "buffer arg_attrs require elem_type and must not use type");
      continue;
    }

    auto type = getStringAttr(dict, "type");
    if (!type || getStringAttr(dict, "elem_type"))
      return kernel.emitOpError(
          "scalar arg_attrs require type and must not use elem_type");
    if (type.getValue() == "f32") {
      if (!formalType.isF32())
        return kernel.emitOpError("scalar f32 arg must have f32 formal type");
    } else if (type.getValue() == "i32" || type.getValue() == "u32") {
      if (!formalType.isSignlessInteger(32))
        return kernel.emitOpError("scalar i32/u32 arg must have i32 formal type");
    } else {
      return kernel.emitOpError("scalar type must be u32, i32, or f32");
    }
  }

  return success();
}

static LogicalResult verifyScheduleShape(KernelOp kernel) {
  if (kernel->getAttr("resource"))
    return kernel.emitOpError(
        "resource metadata is compiler-computed and must not be authored on vc4kernel.kernel");

  ScheduleModeAttr scheduleMode = getKernelScheduleModeAttr(kernel);
  if (!scheduleMode)
    return kernel.emitOpError(
        "schedule_mode must be #vc4kernel.schedule_mode<...>");

  IntegerAttr warpsAttr = getKernelWarpsPerBlockAttr(kernel);
  if (!warpsAttr)
    return kernel.emitOpError("warps_per_block must be an integer attr");
  int64_t warps = warpsAttr.getInt();

  ScheduleMode mode = scheduleMode.getValue();
  if (mode == ScheduleMode::independent_vector && warps != 1)
    return kernel.emitOpError(
        "independent_vector kernels must have warps_per_block = 1");
  if (mode == ScheduleMode::cooperative_block && (warps < 1 || warps > 12))
    return kernel.emitOpError(
        "cooperative_block kernels require warps_per_block in range [1, 12]");
  return success();
}

static bool isVPMUser(Operation *op) {
  return hasName(op, "vc4kernel.vpm_alloc") ||
         hasName(op, "vc4kernel.vpm_write_fragment") ||
         hasName(op, "vc4kernel.vpm_read_fragment") ||
         hasName(op, "vc4kernel.vdr_load_to_vpm") ||
         hasName(op, "vc4kernel.vdw_store_vpm_fragment");
}

static LogicalResult verifyVPMResourceUsage(KernelOp kernel) {
  int64_t totalRows = 0;
  kernel.getBody().walk([&](Operation *op) {
    if (!isVPMUser(op))
      return;
    if (!hasName(op, "vc4kernel.vpm_alloc"))
      return;
    auto rows = op->getAttrOfType<IntegerAttr>("rows");
    if (rows)
      totalRows += rows.getInt();
  });

  if (totalRows > 64)
    return kernel.emitOpError("VPM allocations exceed 64 rows");
  return success();
}

} // namespace

ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symNameAttr;
  if (parser.parseSymbolName(symNameAttr, "sym_name", result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseOptionalRParen())) {
      do {
        OpAsmParser::Argument arg;
        if (parser.parseArgument(arg, /*allowType=*/true))
          return failure();
        entryArgs.push_back(arg);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
    }
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, entryArgs, /*enableNameShadowing=*/true))
    return failure();

  SmallVector<Type, 4> argTypes;
  if (!entryArgs.empty()) {
    argTypes.reserve(entryArgs.size());
    for (const OpAsmParser::Argument &arg : entryArgs)
      argTypes.push_back(arg.type);
  } else if (!body->empty()) {
    argTypes.reserve(body->front().getNumArguments());
    for (BlockArgument arg : body->front().getArguments())
      argTypes.push_back(arg.getType());
  }

  result.attributes.set(
      "function_type",
      TypeAttr::get(FunctionType::get(parser.getContext(), argTypes, {})));
  return success();
}

void KernelOp::print(OpAsmPrinter &p) {
  p << ' ';
  StringAttr symName =
      asStringAttr((*this)->getAttr(SymbolTable::getSymbolAttrName()));
  if (symName)
    p.printSymbolName(symName.getValue());
  else
    p << "<invalid-symbol>";
  Region &body = getBody();
  if (!body.empty() && body.front().getNumArguments() != 0) {
    p << '(';
    llvm::interleaveComma(body.front().getArguments(), p,
                          [&](BlockArgument arg) {
                            p.printOperand(arg);
                            p << " : " << arg.getType();
                          });
    p << ')';
  }
  SmallVector<StringRef, 2> elidedAttrs = {"sym_name", "function_type"};
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

LogicalResult KernelOp::verify() {
  StringAttr symName = getKernelSymNameAttr(*this);
  if (!symName || symName.getValue().empty()) {
    Attribute rawSym = (*this)->getAttr(SymbolTable::getSymbolAttrName());
    if (rawSym)
      return emitOpError("requires non-empty string sym_name; got ")
             << rawSym.getAbstractAttribute().getName();
    return emitOpError("requires non-empty string sym_name");
  }
  StringAttr publicName = getKernelPublicNameAttr(*this);
  if (!publicName || publicName.getValue().empty())
    return emitOpError("public_name must be a non-empty string");
  if (getBody().empty())
    return emitOpError("requires a non-empty body region");
  Block &entry = getBody().front();
  auto functionTypeAttr = (*this)->getAttrOfType<TypeAttr>("function_type");
  auto functionType =
      functionTypeAttr
          ? llvm::dyn_cast<FunctionType>(functionTypeAttr.getValue())
          : FunctionType();
  if (!functionType || functionType.getNumResults() != 0)
    return emitOpError("function_type must be a FunctionType with no results");
  if (functionType.getNumInputs() != entry.getNumArguments())
    return emitOpError(
        "function_type inputs must match kernel entry block argument types");
  for (auto indexed : llvm::enumerate(entry.getArguments())) {
    if (functionType.getInput(indexed.index()) != indexed.value().getType())
      return emitOpError(
          "function_type inputs must match kernel entry block argument types");
  }
  for (BlockArgument arg : entry.getArguments()) {
    if (!arg.getType().isSignlessInteger(32) && !arg.getType().isF32())
      return emitOpError("formal arguments may only be i32 or f32");
  }
  if (failed(verifyArgAttrs(*this, entry)))
    return failure();
  if (failed(verifyScheduleShape(*this)))
    return failure();
  return verifyVPMResourceUsage(*this);
}

LogicalResult ReturnOp::verify() {
  if (!isParentKernel(getOperation()))
    return emitOpError("must appear only inside vc4kernel.kernel");
  if (getOperation()->getNumOperands() != 0)
    return emitOpError("takes no operands");
  return success();
}

LogicalResult ProgramIdOp::verify() {
  return verifyNoUniformIndex(getOperation(), "program_id");
}
LogicalResult WarpIdOp::verify() {
  return verifyNoUniformIndex(getOperation(), "warp_id");
}
LogicalResult LaneRangeOp::verify() {
  return verifyNoUniformIndex(getOperation(), "lane_range");
}

LogicalResult PredFullOp::verify() {
  return verifyPred16(getOperation(), getResult().getType(), "result");
}
LogicalResult PredEmptyOp::verify() {
  return verifyPred16(getOperation(), getResult().getType(), "result");
}
LogicalResult PredTailOp::verify() {
  return verifyPred16(getOperation(), getResult().getType(), "result");
}
LogicalResult PredRectOp::verify() {
  return verifyPred16(getOperation(), getResult().getType(), "result");
}
LogicalResult PredAndOp::verify() {
  if (!isNormalizablePredExpression(getResult()))
    return emitOpError("predicate expression is not normalizable");
  return verifyPred16(getOperation(), getResult().getType(), "result");
}
LogicalResult PredOrOp::verify() {
  if (!isNormalizablePredExpression(getResult()))
    return emitOpError("predicate expression is not normalizable");
  return verifyPred16(getOperation(), getResult().getType(), "result");
}
LogicalResult PredNotOp::verify() {
  if (!isNormalizablePredExpression(getResult()))
    return emitOpError("predicate expression is not normalizable");
  return verifyPred16(getOperation(), getResult().getType(), "result");
}
LogicalResult PredAnyOp::verify() {
  if (!isNormalizablePredExpression(getInput()))
    return emitOpError("predicate expression is not normalizable");
  return success();
}
LogicalResult PredAllOp::verify() {
  if (!isNormalizablePredExpression(getInput()))
    return emitOpError("predicate expression is not normalizable");
  return success();
}

LogicalResult SplatOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  auto vectorType = llvm::dyn_cast<VectorType>(resultType);
  if (!vectorType || !isVC4KernelVector16DataType(resultType))
    return emitTypeError(getOperation(), resultType, "result",
                         "vector<16xi32> or vector<16xf32>");
  if (vectorType.getElementType() != inputType)
    return emitOpError("input scalar type must match result element type");
  return success();
}

LogicalResult FragmentAddOp::verify() {
  if (failed(verifySameType(getOperation(), getLhs().getType(), getRhs().getType(),
                            "fragment_add operands")) ||
      failed(verifySameType(getOperation(), getLhs().getType(),
                            getResult().getType(), "fragment_add result")))
    return failure();
  return verifyVector16Data(getOperation(), getResult().getType(), "result");
}
LogicalResult FragmentSubOp::verify() {
  if (failed(verifySameType(getOperation(), getLhs().getType(), getRhs().getType(),
                            "fragment_sub operands")) ||
      failed(verifySameType(getOperation(), getLhs().getType(),
                            getResult().getType(), "fragment_sub result")))
    return failure();
  return verifyVector16Data(getOperation(), getResult().getType(), "result");
}
LogicalResult FragmentMulOp::verify() {
  if (failed(verifySameType(getOperation(), getLhs().getType(), getRhs().getType(),
                            "fragment_mul operands")) ||
      failed(verifySameType(getOperation(), getLhs().getType(),
                            getResult().getType(), "fragment_mul result")))
    return failure();
  return verifyVector16Data(getOperation(), getResult().getType(), "result");
}
LogicalResult FragmentShlOp::verify() {
  if (!getConstantI32(getAmount()))
    return emitOpError("fragment_shl amount must be a scalar i32 constant");
  return isVC4KernelVector16I32Type(getResult().getType())
             ? success()
             : emitTypeError(getOperation(), getResult().getType(), "result",
                             "vector<16xi32>");
}
LogicalResult FragmentCmpOp::verify() {
  return verifyPred16(getOperation(), getResult().getType(), "result");
}
LogicalResult FragmentSelectOp::verify() {
  if (!isNormalizablePredExpression(getPred()))
    return emitOpError("predicate expression is not normalizable");
  if (failed(verifySameType(getOperation(), getTrueValue().getType(),
                            getFalseValue().getType(),
                            "fragment_select value operands")) ||
      failed(verifySameType(getOperation(), getTrueValue().getType(),
                            getResult().getType(), "fragment_select result")))
    return failure();
  return verifyVector16Data(getOperation(), getResult().getType(), "result");
}
LogicalResult FragmentRotateOp::verify() {
  if (getAmount() < 0 || getAmount() > 15)
    return emitOpError("amount must be in range [0, 15]");
  if (failed(verifySameType(getOperation(), getInput().getType(),
                            getResult().getType(), "fragment_rotate result")))
    return failure();
  return success();
}
LogicalResult FragmentReduceOp::verify() {
  if (getKind() != ReduceKind::add)
    return emitOpError("only #vc4kernel.reduce<add> is supported");
  if (!isNormalizablePredExpression(getPred()))
    return emitOpError("predicate expression is not normalizable");
  if (failed(verifySameType(getOperation(), getInput().getType(),
                            getResult().getType(), "fragment_reduce result")))
    return failure();
  return success();
}

LogicalResult TMULoadFragmentOp::verify() {
  if (!isKnownVectorByteOffsetsAligned4(getByteOffsets()))
    return emitOpError(
        "tmu_load_fragment byte_offsets must be statically 4-byte aligned");
  if (!isNormalizablePredExpression(getPred()))
    return emitOpError("predicate expression is not normalizable");
  return success();
}
LogicalResult VDWStoreFragmentOp::verify() {
  if (!isContiguousByteOffsets(getByteOffsets()))
    return emitOpError("vdw_store_fragment requires contiguous byte offsets");
  if (!isKnownVectorByteOffsetsAligned4(getByteOffsets()))
    return emitOpError(
        "vdw_store_fragment byte_offsets must be statically 4-byte aligned");
  if (!isNormalizablePredExpression(getPred()))
    return emitOpError("predicate expression is not normalizable");
  return success();
}
LogicalResult VPMAllocOp::verify() {
  if (getRows() < 1 || getRows() > 64)
    return emitOpError("rows must be in range [1, 64]");
  if (getElemBytes() != 4)
    return emitOpError("elem_bytes must be 4");
  return success();
}
LogicalResult VPMWriteFragmentOp::verify() {
  if (failed(verifyOrientationRowOnly(getOperation())) ||
      failed(verifyVPMRowInBounds(getOperation(), getTile(), getRow())))
    return failure();
  if (!isNormalizablePredExpression(getPred()))
    return emitOpError("predicate expression is not normalizable");
  return success();
}
LogicalResult VPMReadFragmentOp::verify() {
  if (failed(verifyOrientationRowOnly(getOperation())) ||
      failed(verifyVPMRowInBounds(getOperation(), getTile(), getRow())))
    return failure();
  if (!isNormalizablePredExpression(getPred()))
    return emitOpError("predicate expression is not normalizable");
  return success();
}
LogicalResult VDRLoadToVPMOp::verify() {
  if (getOperation()->getNumOperands() != 4)
    return emitOpError("does not accept a predicate operand");
  if (!isKnownScalarByteOffsetAligned4(getByteOffset()))
    return emitOpError(
        "vdr_load_to_vpm byte_offset must be statically 4-byte aligned");
  if (getRows() <= 0)
    return emitOpError("rows must be positive");
  if (getCols() < 1 || getCols() > 16)
    return emitOpError("cols must be in range [1, 16]");
  if (getElemBytes() != 4)
    return emitOpError("elem_bytes must be 4");
  if (getGlobalStrideBytes() <= 0 || getGlobalStrideBytes() % 4 != 0)
    return emitOpError("global_stride_bytes must be positive and 4-byte aligned");
  return verifyVPMRowInBounds(getOperation(), getTile(), getDstRow(), getRows());
}
LogicalResult VDWStoreVPMFragmentOp::verify() {
  if (getElemBytes() != 4)
    return emitOpError("elem_bytes must be 4");
  if (!isKnownScalarByteOffsetAligned4(getByteOffset()))
    return emitOpError(
        "vdw_store_vpm_fragment byte_offset must be statically 4-byte aligned");
  if (!isNormalizablePredExpression(getPred()))
    return emitOpError("predicate expression is not normalizable");
  return verifyVPMRowInBounds(getOperation(), getTile(), getSrcRow());
}
LogicalResult BarrierOp::verify() {
  KernelOp kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError("must appear only inside vc4kernel.kernel");
  ScheduleModeAttr scheduleMode = getKernelScheduleModeAttr(kernel);
  if (!scheduleMode)
    return emitOpError("requires verified kernel schedule metadata");
  if (scheduleMode.getValue() != ScheduleMode::cooperative_block)
    return emitOpError("barrier requires cooperative_block schedule_mode");
  return success();
}

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelOps.cpp.inc"
