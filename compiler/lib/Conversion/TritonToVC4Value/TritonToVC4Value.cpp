//===- TritonToVC4Value.cpp - TTIR to VC4 value lowering ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the first durable C++ frontend boundary for real Triton
// TTIR.  It deliberately has a different shape from the Phase 7 Python smoke
// importer:
//
//   * The input must already be parsed as MLIR with Triton's `tt` dialect
//     registered by the optional frontend tool.
//   * Classification is structural: operation names, SSA use-defs, attributes,
//     and MLIR types drive lowering.  Kernel names are not semantic.
//   * The output is only the VC4 standard value layer:
//       func + vc4value + vector + memref + arith/math/scf/cf.
//   * No vc4kernel/ssavc4/scheduled-vc4 operations are emitted here.
//   * Unsupported TTIR forms are classified as staged Phase 7.5 limitations
//     unless the target profile has a permanent reject proof.
//
// The initial executable subset is intentionally the Phase 7 elementwise V1
// subset that is already hardware-proven through the old importer:
//
//   tt.get_program_id axis 0
//   tt.make_range 0..16
//   contiguous base + pid*16 + arange pointer expressions
//   tail mask offsets < n
//   masked tt.load with other=0
//   masked tt.store with the same tail mask
//   straight-line i32/f32 add/sub/mul/cmp/select compute DAGs
//
// Future phases should extend the semantic planners below (launch/range/pointer/
// mask/memory/compute/reduction/dot) instead of adding kernel-name templates.
//
//===----------------------------------------------------------------------===//

#include "vc4/Conversion/TritonToVC4Value/TritonToVC4Value.h"

#include "vc4/Dialect/VC4Value/IR/VC4ValueAttrs.h"
#include "vc4/Dialect/VC4Value/IR/VC4ValueDialect.h"
#include "vc4/Dialect/VC4Value/IR/VC4ValueOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kTTFuncOpName("tt.func");
constexpr llvm::StringLiteral kTTReturnOpName("tt.return");
constexpr llvm::StringLiteral kTTGetProgramIdOpName("tt.get_program_id");
constexpr llvm::StringLiteral kTTGetNumProgramsOpName("tt.get_num_programs");
constexpr llvm::StringLiteral kTTMakeRangeOpName("tt.make_range");
constexpr llvm::StringLiteral kTTSplatOpName("tt.splat");
constexpr llvm::StringLiteral kTTBroadcastOpName("tt.broadcast");
constexpr llvm::StringLiteral kTTAddPtrOpName("tt.addptr");
constexpr llvm::StringLiteral kTTLoadOpName("tt.load");
constexpr llvm::StringLiteral kTTStoreOpName("tt.store");
constexpr llvm::StringLiteral kTTDotOpName("tt.dot");
constexpr llvm::StringLiteral kTTReduceOpName("tt.reduce");
constexpr llvm::StringLiteral kTTReduceReturnOpName("tt.reduce.return");
constexpr llvm::StringLiteral kTTExpandDimsOpName("tt.expand_dims");
constexpr llvm::StringLiteral kTTTransOpName("tt.trans");
constexpr llvm::StringLiteral kTTAdvanceOpName("tt.advance");

constexpr llvm::StringLiteral kVC4ValueKernelAttr("vc4value.kernel");
constexpr llvm::StringLiteral kVC4ValueGridRankAttr("vc4value.grid_rank");
constexpr llvm::StringLiteral kVC4ValueArgNameAttr("vc4value.arg_name");
constexpr llvm::StringLiteral kVC4ValueDirectionAttr("vc4value.direction");
constexpr llvm::StringLiteral kVC4ValueShapeArgsAttr("vc4value.shape_args");
constexpr llvm::StringLiteral kVC4ValueFPDomainAttr("vc4value.fp_domain");

constexpr llvm::StringLiteral kVC4ValueProgramIdOpName("vc4value.program_id");
constexpr llvm::StringLiteral kVC4ValueNumProgramsOpName("vc4value.num_programs");

constexpr unsigned kPhase7VectorWidth = 16;

static bool hasName(Operation *op, StringRef name) {
  return op && op->getName().getStringRef() == name;
}

static std::string stringify(Attribute attr) {
  std::string out;
  llvm::raw_string_ostream os(out);
  if (attr)
    attr.print(os);
  return out;
}

static LogicalResult emitStagedDiagnostic(Operation *op, const Twine &detail) {
  return op->emitOpError()
         << detail << " is not Phase 7.5 C++ TTIR-to-VC4Value lowerable; "
         << "staged TTIR target-profile feature. READY_FOR_TRITON remains NO";
}

static LogicalResult emitPermanentReject(Operation *op, const Twine &detail) {
  return op->emitOpError()
         << detail << " is rejected by the VC4 TTIR target profile. "
         << "READY_FOR_TRITON remains NO";
}

static LogicalResult emitInternalError(Operation *op, const Twine &detail) {
  return op->emitError() << "internal TTIR-to-VC4Value lowering error: "
                         << detail;
}

static bool isTritonDialectOp(Operation *op) {
  return op && op->getName().getDialectNamespace() == "tt";
}

static bool isForbiddenProducerOrBackendDialect(Operation *op) {
  if (!op)
    return false;
  StringRef dialect = op->getName().getDialectNamespace();
  return dialect == "ttg" || dialect == "triton_gpu" || dialect == "nvgpu" ||
         dialect == "nvvm" || dialect == "rocdl" || dialect == "gpu" ||
         dialect == "llvm" || dialect == "spirv";
}

static bool isScalarI32(Type type) { return type.isSignlessInteger(32); }
static bool isScalarI64(Type type) { return type.isSignlessInteger(64); }
static bool isScalarF32(Type type) { return type.isF32(); }

static bool isRankedTensor(Type type, unsigned rank, int64_t dim0) {
  auto shaped = llvm::dyn_cast<RankedTensorType>(type);
  return shaped && shaped.getRank() == static_cast<int64_t>(rank) &&
         shaped.getDimSize(0) == dim0;
}

static bool isTensor16(Type type) {
  return isRankedTensor(type, /*rank=*/1, kPhase7VectorWidth);
}

static bool isTensor16I1(Type type) {
  auto shaped = llvm::dyn_cast<RankedTensorType>(type);
  return shaped && shaped.getRank() == 1 && shaped.getDimSize(0) == 16 &&
         shaped.getElementType().isInteger(1);
}

static bool isTensor16I32(Type type) {
  auto shaped = llvm::dyn_cast<RankedTensorType>(type);
  return shaped && shaped.getRank() == 1 && shaped.getDimSize(0) == 16 &&
         shaped.getElementType().isSignlessInteger(32);
}

static bool isTensor16F32(Type type) {
  auto shaped = llvm::dyn_cast<RankedTensorType>(type);
  return shaped && shaped.getRank() == 1 && shaped.getDimSize(0) == 16 &&
         shaped.getElementType().isF32();
}

static bool containsRankGreaterThanOneTensor(Type type) {
  auto shaped = llvm::dyn_cast<RankedTensorType>(type);
  return shaped && shaped.getRank() > 1;
}

enum class TTIRScalarElementKind { I1, I32, F32, Unsupported };

struct TTIRPointerTypeInfo {
  Type pointeeType;
  TTIRScalarElementKind pointeeKind = TTIRScalarElementKind::Unsupported;
  std::optional<unsigned> addressSpace;
  bool isSupportedPhase75Element = false;
  bool isTensorPointer = false;
};

struct TTIRTensorTypeInfo {
  RankedTensorType tensorType;
  int64_t rank = 0;
  SmallVector<int64_t, 4> shape;
  Type elementType;
  TTIRScalarElementKind elementKind = TTIRScalarElementKind::Unsupported;
};

class TTIRTypeAdapter {
public:
  explicit TTIRTypeAdapter(MLIRContext *ctx) : ctx(ctx) {}

  std::optional<TTIRPointerTypeInfo> classifyPointerType(Type type) const {
    auto pointerType = llvm::dyn_cast<mlir::triton::PointerType>(type);
    if (!pointerType)
      return std::nullopt;

    TTIRPointerTypeInfo info;
    info.pointeeType = pointerType.getPointeeType();
    info.addressSpace = static_cast<unsigned>(pointerType.getAddressSpace());
    info.isTensorPointer = llvm::isa<RankedTensorType>(info.pointeeType);
    Type elementType = info.pointeeType;
    if (auto tensor = llvm::dyn_cast<RankedTensorType>(info.pointeeType))
      elementType = tensor.getElementType();
    info.pointeeKind = classifyScalarElement(elementType);
    info.isSupportedPhase75Element =
        !info.isTensorPointer && (info.pointeeKind == TTIRScalarElementKind::I32 ||
                                  info.pointeeKind == TTIRScalarElementKind::F32);
    return info;
  }

  std::optional<TTIRPointerTypeInfo>
  classifyTensorPointerElement(Type type) const {
    auto shaped = llvm::dyn_cast<RankedTensorType>(type);
    if (!shaped)
      return std::nullopt;
    return classifyPointerType(shaped.getElementType());
  }

  std::optional<TTIRTensorTypeInfo> classifyRankedTensor(Type type) const {
    auto shaped = llvm::dyn_cast<RankedTensorType>(type);
    if (!shaped)
      return std::nullopt;

    TTIRTensorTypeInfo info;
    info.tensorType = shaped;
    info.rank = shaped.getRank();
    info.shape.append(shaped.getShape().begin(), shaped.getShape().end());
    info.elementType = shaped.getElementType();
    info.elementKind = classifyScalarElement(info.elementType);
    return info;
  }

  Type convertTensorToValueVector(Type type, OpBuilder &builder) const {
    std::optional<TTIRTensorTypeInfo> tensor = classifyRankedTensor(type);
    if (!tensor || tensor->rank != 1 || tensor->shape[0] != kPhase7VectorWidth)
      return {};
    Type elem = getValueElementType(tensor->elementKind, builder);
    if (!elem)
      return {};
    return VectorType::get({kPhase7VectorWidth}, elem);
  }

  Type getValueElementType(TTIRScalarElementKind kind, OpBuilder &builder) const {
    switch (kind) {
    case TTIRScalarElementKind::I1:
      return builder.getI1Type();
    case TTIRScalarElementKind::I32:
      return builder.getI32Type();
    case TTIRScalarElementKind::F32:
      return builder.getF32Type();
    case TTIRScalarElementKind::Unsupported:
      return {};
    }
    return {};
  }

private:
  TTIRScalarElementKind classifyScalarElement(Type type) const {
    if (type.isInteger(1))
      return TTIRScalarElementKind::I1;
    if (type.isSignlessInteger(32))
      return TTIRScalarElementKind::I32;
    if (type.isF32())
      return TTIRScalarElementKind::F32;
    return TTIRScalarElementKind::Unsupported;
  }

  MLIRContext *ctx;
};

static Type convertScalarArgType(Type type, bool isSizeLike, OpBuilder &builder) {
  if (isSizeLike && isScalarI32(type))
    return builder.getIndexType();
  if (isScalarI32(type) || isScalarF32(type))
    return type;
  return {};
}

static FailureOr<Attribute> retargetDenseAttr(DenseElementsAttr dense,
                                              ShapedType newType) {
  if (newType.getNumElements() != dense.getNumElements())
    return failure();
  if (newType.getElementType().isSignlessInteger(32) ||
      newType.getElementType().isInteger(1)) {
    SmallVector<APInt, 16> values;
    values.reserve(dense.getNumElements());
    for (APInt value : dense.getValues<APInt>())
      values.push_back(value);
    return DenseIntElementsAttr::get(newType, values);
  }
  if (newType.getElementType().isF32()) {
    SmallVector<APFloat, 16> values;
    values.reserve(dense.getNumElements());
    for (APFloat value : dense.getValues<APFloat>())
      values.push_back(value);
    return DenseFPElementsAttr::get(newType, values);
  }
  return failure();
}

static std::string sanitizeSymbolName(StringRef name) {
  std::string result;
  result.reserve(name.size());
  for (char c : name) {
    unsigned char uc = static_cast<unsigned char>(c);
    result.push_back(std::isalnum(uc) || c == '_' ? c : '_');
  }
  if (result.empty() ||
      !(std::isalpha(static_cast<unsigned char>(result.front())) ||
        result.front() == '_'))
    result.insert(result.begin(), '_');
  return result;
}

static void collectNameLocStrings(Location loc,
                                  SmallVectorImpl<StringRef> &names) {
  if (auto name = llvm::dyn_cast<NameLoc>(loc)) {
    names.push_back(name.getName().getValue());
    collectNameLocStrings(name.getChildLoc(), names);
    return;
  }
  if (auto call = llvm::dyn_cast<CallSiteLoc>(loc)) {
    collectNameLocStrings(call.getCallee(), names);
    collectNameLocStrings(call.getCaller(), names);
    return;
  }
  if (auto fused = llvm::dyn_cast<FusedLoc>(loc)) {
    for (Location child : fused.getLocations())
      collectNameLocStrings(child, names);
    return;
  }
}

static std::string inferSourceArgName(BlockArgument arg, unsigned index) {
  SmallVector<StringRef, 4> names;
  collectNameLocStrings(arg.getLoc(), names);
  for (StringRef name : names) {
    if (name.empty())
      continue;
    std::string cleaned = name.str();
    if (llvm::StringRef(cleaned).ends_with("_ptr"))
      cleaned.resize(cleaned.size() - 4);
    if (cleaned == "n_elements")
      cleaned = "n";
    if (!cleaned.empty())
      return sanitizeSymbolName(cleaned);
  }
  return ("arg" + Twine(index)).str();
}

static bool isLikelySizeArgName(StringRef name) {
  return name == "n" || name == "n_elements" || name == "size" ||
         name == "numel" || name.ends_with("_n") || name.ends_with("_size") ||
         name.ends_with("_elements");
}

static bool hasAnyName(Operation *op, ArrayRef<StringRef> names) {
  StringRef name = op->getName().getStringRef();
  return llvm::is_contained(names, name);
}

static std::optional<int64_t> getIntegerAttrValue(Operation *op,
                                                  StringRef name) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return std::nullopt;
}

static std::string getAttrString(Operation *op, StringRef name) {
  if (Attribute attr = op->getAttr(name))
    return stringify(attr);
  return "";
}

static bool isAxisZero(Operation *op) {
  std::string axis = getAttrString(op, "axis");
  // Triton prints the axis in custom form (`x`, `y`, `z`) but the generic attr
  // may be an enum attr.  Accept only axis-0 spellings.  Axis 1/2 remain staged
  // until the value launch/grid lowering phases expand beyond V1.
  return axis == "#tt.program_id_dim<x>" || axis == "x" || axis == "0" ||
         axis.find("x") != std::string::npos || axis.find("0") != std::string::npos;
}

static bool isMakeRange0To16(Operation *op) {
  std::optional<int64_t> start = getIntegerAttrValue(op, "start");
  std::optional<int64_t> end = getIntegerAttrValue(op, "end");
  if (start && end)
    return *start == 0 && *end == 16;
  std::string text;
  llvm::raw_string_ostream os(text);
  op->print(os, OpPrintingFlags().skipRegions());
  return StringRef(text).contains("start = 0") &&
         StringRef(text).contains("end = 16");
}

static bool isZeroLikeConstant(Operation *op) {
  auto constant = llvm::dyn_cast_or_null<arith::ConstantOp>(op);
  if (!constant)
    return false;
  Attribute value = constant.getValue();
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(value))
    return intAttr.getValue().isZero();
  if (auto floatAttr = llvm::dyn_cast<FloatAttr>(value))
    return floatAttr.getValue().isZero();
  if (auto dense = llvm::dyn_cast<DenseElementsAttr>(value)) {
    if (llvm::isa<FloatType>(dense.getElementType())) {
      for (APFloat fp : dense.getValues<APFloat>())
        if (!fp.isZero())
          return false;
      return true;
    }
    if (dense.getElementType().isIntOrIndex()) {
      for (APInt value : dense.getValues<APInt>())
        if (!value.isZero())
          return false;
      return true;
    }
  }
  return false;
}

static AffineMap getRank1IdentityMap(MLIRContext *ctx) {
  Builder b(ctx);
  return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, b.getAffineDimExpr(0));
}

static Operation *createGenericOp(OpBuilder &builder, Location loc,
                                  StringRef name, ValueRange operands,
                                  ArrayRef<NamedAttribute> attrs,
                                  TypeRange resultTypes = {}) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  for (NamedAttribute attr : attrs)
    state.addAttribute(attr.getName(), attr.getValue());
  return builder.create(state);
}

static Value createGenericOpWithResult(OpBuilder &builder, Location loc,
                                       StringRef name, ValueRange operands,
                                       ArrayRef<NamedAttribute> attrs,
                                       Type resultType) {
  Operation *op = createGenericOp(builder, loc, name, operands, attrs, resultType);
  return op->getResult(0);
}

static Value createVC4ValueProgramId(OpBuilder &builder, Location loc,
                                     int64_t axis) {
  return createGenericOpWithResult(
      builder, loc, kVC4ValueProgramIdOpName, {},
      {builder.getNamedAttr("axis", builder.getI32IntegerAttr(axis))},
      builder.getIndexType());
}

static Value createVectorStep(OpBuilder &builder, Location loc) {
  return createGenericOpWithResult(
      builder, loc, "vector.step", {}, {},
      VectorType::get({16}, builder.getIndexType()));
}

static Value createVectorBroadcast(OpBuilder &builder, Location loc, Value value,
                                   Type resultType) {
  return createGenericOpWithResult(builder, loc, "vector.broadcast", value, {},
                                   resultType);
}

static Value createVectorCreateMask(OpBuilder &builder, Location loc,
                                    Value activeCount) {
  return createGenericOpWithResult(
      builder, loc, "vector.create_mask", activeCount, {},
      VectorType::get({16}, builder.getI1Type()));
}

static Value createVectorTransferRead(OpBuilder &builder, Location loc,
                                      Value source, Value index, Value padding,
                                      Value mask, VectorType resultType) {
  MLIRContext *ctx = builder.getContext();
  SmallVector<NamedAttribute, 4> attrs;
  attrs.push_back(builder.getNamedAttr(
      "permutation_map", AffineMapAttr::get(getRank1IdentityMap(ctx))));
  attrs.push_back(builder.getNamedAttr(
      "in_bounds", builder.getArrayAttr({builder.getBoolAttr(false)})));
  attrs.push_back(builder.getNamedAttr(
      "operandSegmentSizes", DenseI32ArrayAttr::get(ctx, {1, 1, 1, 1})));
  return createGenericOpWithResult(builder, loc, "vector.transfer_read",
                                   {source, index, padding, mask}, attrs,
                                   resultType);
}

static void createVectorTransferWrite(OpBuilder &builder, Location loc,
                                      Value value, Value dest, Value index,
                                      Value mask) {
  MLIRContext *ctx = builder.getContext();
  SmallVector<NamedAttribute, 4> attrs;
  attrs.push_back(builder.getNamedAttr(
      "permutation_map", AffineMapAttr::get(getRank1IdentityMap(ctx))));
  attrs.push_back(builder.getNamedAttr(
      "in_bounds", builder.getArrayAttr({builder.getBoolAttr(false)})));
  attrs.push_back(builder.getNamedAttr(
      "operandSegmentSizes", DenseI32ArrayAttr::get(ctx, {1, 1, 1, 1})));
  createGenericOp(builder, loc, "vector.transfer_write", {value, dest, index, mask},
                  attrs);
}

struct TTIRArgInfo {
  BlockArgument sourceArg;
  std::string valueName;
  Type sourceType;
  Type valueType;
  bool isPointer = false;
  bool isSizeLike = false;
  TTIRScalarElementKind pointerElement = TTIRScalarElementKind::Unsupported;
  bool loaded = false;
  bool stored = false;
};

struct PointerExpr {
  BlockArgument sourcePointerArg;
  Value valueMemref;
  Value offsetValue;
  TTIRScalarElementKind element = TTIRScalarElementKind::Unsupported;
  bool hasCanonicalOffset = false;
};

struct CommonPlan {
  Operation *programId = nullptr;
  Operation *makeRange = nullptr;
  Value offsetValue;
  Value tailMaskValue;
  BlockArgument sizeArg;
};

class FunctionPlanner {
public:
  explicit FunctionPlanner(Operation *funcOp)
      : funcOp(funcOp), typeAdapter(funcOp->getContext()) {}

  LogicalResult analyze() {
    if (!hasName(funcOp, kTTFuncOpName))
      return funcOp->emitOpError("expected tt.func");
    if (auto fn = llvm::dyn_cast<FunctionOpInterface>(funcOp)) {
      auto fnType = llvm::dyn_cast<FunctionType>(fn.getFunctionType());
      if (fnType && !fnType.getResults().empty())
        return emitStagedDiagnostic(funcOp, "TTIR functions with results");
    } else {
      return funcOp->emitOpError("expected tt.func to implement FunctionOpInterface");
    }
    if (funcOp->getNumRegions() != 1 || funcOp->getRegion(0).empty())
      return emitStagedDiagnostic(funcOp, "tt.func without a single body region");
    if (!llvm::hasSingleElement(funcOp->getRegion(0)))
      return emitStagedDiagnostic(funcOp, "multi-block TTIR functions");

    Block &entry = funcOp->getRegion(0).front();
    if (entry.empty())
      return emitStagedDiagnostic(funcOp, "empty TTIR function body");

    if (failed(collectArgs(entry)))
      return failure();
    if (failed(scanOps(entry)))
      return failure();
    if (failed(classifyCommonPlan()))
      return failure();
    if (failed(classifyMemoryDirections()))
      return failure();
    markRequiredDataflow();
    return success();
  }

  Operation *getFuncOp() const { return funcOp; }
  Block &getEntryBlock() const { return funcOp->getRegion(0).front(); }
  ArrayRef<TTIRArgInfo> getArgs() const { return args; }
  const CommonPlan &getCommonPlan() const { return common; }

  TTIRArgInfo *lookupArg(BlockArgument arg) {
    auto it = argIndex.find(arg);
    if (it == argIndex.end())
      return nullptr;
    return &args[it->second];
  }

  Operation *lookupDef(Value value) const {
    return value.getDefiningOp();
  }

  bool isCanonicalOffsetValue(Value value) const { return value == common.offsetValue; }
  bool isTailMaskValue(Value value) const { return value == common.tailMaskValue; }
  bool isRequiredDataValue(Value value) const { return requiredDataValues.contains(value); }

  bool isCanonicalInfrastructure(Operation *op) const {
    if (op == common.programId || op == common.makeRange)
      return true;
    if (op->getNumResults() == 0)
      return false;
    for (Value result : op->getResults()) {
      if (result == common.offsetValue || result == common.tailMaskValue)
        return true;
      if (canonicalInfrastructureValues.contains(result))
        return true;
    }
    return false;
  }

  FailureOr<PointerExpr> classifyPointer(Value ptrValue) const {
    if (auto found = pointerExprs.find(ptrValue); found != pointerExprs.end())
      return found->second;

    Operation *def = ptrValue.getDefiningOp();
    if (!def)
      return failure();

    if (hasName(def, kTTAddPtrOpName)) {
      if (def->getNumOperands() != 2)
        return def->emitOpError("expected tt.addptr with pointer and offset");
      FailureOr<PointerExpr> base = classifyPointer(def->getOperand(0));
      if (failed(base))
        return failure();
      if (def->getOperand(1) != common.offsetValue)
        return emitStagedDiagnostic(def, "non-canonical pointer offset");
      PointerExpr expr = *base;
      expr.offsetValue = def->getOperand(1);
      expr.hasCanonicalOffset = true;
      return expr;
    }

    if (hasName(def, kTTSplatOpName) && def->getNumOperands() == 1) {
      Value scalar = def->getOperand(0);
      auto blockArg = llvm::dyn_cast<BlockArgument>(scalar);
      if (!blockArg)
        return emitStagedDiagnostic(def, "pointer base not formed from scalar pointer argument splat");
      auto it = argIndex.find(blockArg);
      if (it == argIndex.end() || !args[it->second].isPointer)
        return emitStagedDiagnostic(def, "tt.splat pointer base not backed by a pointer argument");
      PointerExpr expr;
      expr.sourcePointerArg = blockArg;
      expr.element = args[it->second].pointerElement;
      return expr;
    }

    return emitStagedDiagnostic(def, "unsupported pointer expression");
  }

private:
  LogicalResult collectArgs(Block &entry) {
    args.reserve(entry.getNumArguments());
    for (auto [index, arg] : llvm::enumerate(entry.getArguments())) {
      TTIRArgInfo info;
      info.sourceArg = arg;
      info.sourceType = arg.getType();
      info.valueName = inferSourceArgName(arg, index);
      if (auto pointer = typeAdapter.classifyPointerType(arg.getType())) {
        if (pointer->isTensorPointer)
          return emitStagedDiagnostic(funcOp, "block pointer or tensor pointer argument");
        if (!pointer->isSupportedPhase75Element)
          return emitStagedDiagnostic(funcOp, "unsupported pointer argument element type");
        info.isPointer = true;
        info.pointerElement = pointer->pointeeKind;
      } else if (isScalarI32(arg.getType()) || isScalarF32(arg.getType())) {
        info.isSizeLike = isLikelySizeArgName(info.valueName);
      } else {
        if (typeAdapter.classifyTensorPointerElement(arg.getType()))
          return emitStagedDiagnostic(funcOp, "unsupported pointer argument element type");
        return emitStagedDiagnostic(funcOp, "unsupported TTIR function argument type");
      }
      argIndex[arg] = args.size();
      args.push_back(info);
    }
    return success();
  }

  LogicalResult scanOps(Block &entry) {
    // Prefer high-level staged feature diagnostics before more generic tensor
    // shape diagnostics.  This keeps future dot/reduction tests from being
    // masked by their rank-2 operands or helper constants.
    for (Operation &op : entry.getOperations()) {
      if (hasName(&op, kTTDotOpName))
        return emitStagedDiagnostic(&op, "tt.dot / contract");
      if (hasName(&op, kTTReduceOpName) || hasName(&op, kTTReduceReturnOpName))
        return emitStagedDiagnostic(&op, "tt.reduce");
    }

    for (Operation &op : entry.getOperations()) {
      if (isForbiddenProducerOrBackendDialect(&op))
        return emitPermanentReject(&op, "backend/non-TTIR dialect operation in TTIR input");
      if (hasName(&op, kTTBroadcastOpName) || hasName(&op, kTTExpandDimsOpName) ||
          hasName(&op, kTTTransOpName))
        return emitStagedDiagnostic(&op, "rank-2/shape-changing TTIR tensor form");
      for (Type type : op.getResultTypes())
        if (containsRankGreaterThanOneTensor(type))
          return emitStagedDiagnostic(&op, "rank-2 or higher TTIR tensor result");
    }
    return success();
  }

  LogicalResult classifyCommonPlan() {
    Block &entry = getEntryBlock();
    SmallVector<Operation *, 2> programIds;
    SmallVector<Operation *, 2> ranges;
    SmallVector<Operation *, 2> stores;
    SmallVector<Operation *, 4> addptrs;
    for (Operation &op : entry) {
      if (hasName(&op, kTTGetProgramIdOpName))
        programIds.push_back(&op);
      if (hasName(&op, kTTMakeRangeOpName))
        ranges.push_back(&op);
      if (hasName(&op, kTTStoreOpName))
        stores.push_back(&op);
      if (hasName(&op, kTTAddPtrOpName))
        addptrs.push_back(&op);
    }

    if (programIds.size() != 1)
      return emitStagedDiagnostic(funcOp, "TTIR functions with other than one program_id");
    if (!isAxisZero(programIds.front()))
      return emitStagedDiagnostic(programIds.front(), "program_id axis other than 0/x");
    common.programId = programIds.front();

    if (ranges.size() != 1 || !isMakeRange0To16(ranges.front()))
      return emitStagedDiagnostic(funcOp, "tt.make_range other than 0..16");
    common.makeRange = ranges.front();

    if (stores.empty())
      return emitStagedDiagnostic(funcOp, "TTIR elementwise V1 without store");
    if (addptrs.empty())
      return emitStagedDiagnostic(funcOp, "TTIR elementwise V1 without addptr");

    Value offset;
    for (Operation *addptr : addptrs) {
      if (addptr->getNumOperands() != 2)
        return emitStagedDiagnostic(addptr, "tt.addptr without pointer and offset operands");
      if (!offset)
        offset = addptr->getOperand(1);
      else if (offset != addptr->getOperand(1))
        return emitStagedDiagnostic(addptr, "multiple distinct pointer offset expressions");
    }
    if (!offset)
      return emitInternalError(funcOp, "missing offset after addptr scan");
    if (failed(verifyCanonicalOffset(offset)))
      return failure();
    common.offsetValue = offset;

    Value mask;
    for (Operation *store : stores) {
      if (store->getNumOperands() != 3)
        return emitStagedDiagnostic(store, "tt.store without canonical mask operand");
      if (!mask)
        mask = store->getOperand(2);
      else if (mask != store->getOperand(2))
        return emitStagedDiagnostic(store, "multiple distinct store masks");
    }
    if (!mask)
      return emitInternalError(funcOp, "missing tail mask after store scan");
    if (failed(verifyCanonicalTailMask(mask, offset)))
      return failure();
    common.tailMaskValue = mask;
    return success();
  }

  LogicalResult verifyCanonicalOffset(Value offset) {
    Operation *add = offset.getDefiningOp();
    if (!hasName(add, "arith.addi") || add->getNumOperands() != 2)
      return emitStagedDiagnostic(add ? add : funcOp,
                                  "pointer offset not formed by arith.addi");

    Value rangeResult = common.makeRange->getResult(0);
    Value other;
    if (add->getOperand(0) == rangeResult)
      other = add->getOperand(1);
    else if (add->getOperand(1) == rangeResult)
      other = add->getOperand(0);
    else
      return emitStagedDiagnostic(add, "pointer offset missing tt.make_range lanes");

    Operation *splat = other.getDefiningOp();
    if (!hasName(splat, kTTSplatOpName) || splat->getNumOperands() != 1)
      return emitStagedDiagnostic(add, "pointer offset base not formed by tt.splat");

    Operation *mul = splat->getOperand(0).getDefiningOp();
    if (!hasName(mul, "arith.muli") || mul->getNumOperands() != 2)
      return emitStagedDiagnostic(splat, "pointer offset base not formed by pid * 16");

    if (mul->getOperand(0) != common.programId->getResult(0) &&
        mul->getOperand(1) != common.programId->getResult(0))
      return emitStagedDiagnostic(mul, "pointer offset base missing program_id");

    Value factor = mul->getOperand(0) == common.programId->getResult(0)
                       ? mul->getOperand(1)
                       : mul->getOperand(0);
    if (auto cst = factor.getDefiningOp<arith::ConstantOp>()) {
      if (auto integer = llvm::dyn_cast<IntegerAttr>(cst.getValue())) {
        if (integer.getInt() == 16) {
          canonicalInfrastructureValues.insert(factor);
          canonicalInfrastructureValues.insert(splat->getOperand(0));
          canonicalInfrastructureValues.insert(other);
          canonicalInfrastructureValues.insert(offset);
          return success();
        }
      }
    }
    return emitStagedDiagnostic(mul, "pointer offset base factor other than 16");
  }

  LogicalResult verifyCanonicalTailMask(Value mask, Value offset) {
    Operation *cmp = mask.getDefiningOp();
    auto cmpi = llvm::dyn_cast_or_null<arith::CmpIOp>(cmp);
    if (!cmpi)
      return emitStagedDiagnostic(cmp ? cmp : funcOp,
                                  "tail mask not formed by arith.cmpi");
    if (cmpi.getPredicate() != arith::CmpIPredicate::slt)
      return emitStagedDiagnostic(cmp, "tail mask predicate other than slt");
    if (cmpi.getOperand(0) != offset)
      return emitStagedDiagnostic(cmp, "tail mask left operand not canonical offset");

    Operation *rhsSplat = cmpi.getOperand(1).getDefiningOp();
    if (!hasName(rhsSplat, kTTSplatOpName) || rhsSplat->getNumOperands() != 1)
      return emitStagedDiagnostic(cmp, "tail mask right operand not scalar size splat");
    auto sizeArg = llvm::dyn_cast<BlockArgument>(rhsSplat->getOperand(0));
    if (!sizeArg)
      return emitStagedDiagnostic(rhsSplat, "tail mask size is not a function argument");
    auto it = argIndex.find(sizeArg);
    if (it == argIndex.end() || !args[it->second].isSizeLike)
      return emitStagedDiagnostic(rhsSplat, "tail mask size argument is not ABI size-like");
    common.sizeArg = sizeArg;
    canonicalInfrastructureValues.insert(rhsSplat->getResult(0));
    canonicalInfrastructureValues.insert(mask);
    return success();
  }

  LogicalResult classifyMemoryDirections() {
    for (Operation &op : getEntryBlock()) {
      if (!hasName(&op, kTTLoadOpName) && !hasName(&op, kTTStoreOpName))
        continue;
      if (op.getNumOperands() < 1)
        return emitStagedDiagnostic(&op, "memory op without pointer operand");
      FailureOr<PointerExpr> ptr = classifyPointer(op.getOperand(0));
      if (failed(ptr))
        return failure();
      auto it = argIndex.find(ptr->sourcePointerArg);
      if (it == argIndex.end())
        return emitInternalError(&op, "pointer base argument missing from arg table");
      if (hasName(&op, kTTLoadOpName))
        args[it->second].loaded = true;
      if (hasName(&op, kTTStoreOpName))
        args[it->second].stored = true;
    }
    return success();
  }

  void markRequiredValue(Value value) {
    if (!value || requiredDataValues.contains(value))
      return;
    requiredDataValues.insert(value);
    if (llvm::isa<BlockArgument>(value))
      return;
    Operation *def = value.getDefiningOp();
    if (!def)
      return;

    // Pointer, mask, and offset infrastructure are planned separately.
    if (hasName(def, kTTAddPtrOpName) || hasName(def, kTTMakeRangeOpName) ||
        hasName(def, kTTGetProgramIdOpName))
      return;
    if (def->getNumResults() == 1 &&
        (def->getResult(0) == common.offsetValue ||
         def->getResult(0) == common.tailMaskValue))
      return;

    if (hasName(def, kTTLoadOpName))
      return;

    for (Value operand : def->getOperands())
      markRequiredValue(operand);
  }

  void markRequiredDataflow() {
    for (Operation &op : getEntryBlock())
      if (hasName(&op, kTTStoreOpName) && op.getNumOperands() >= 2)
        markRequiredValue(op.getOperand(1));
  }

  Operation *funcOp;
  TTIRTypeAdapter typeAdapter;
  SmallVector<TTIRArgInfo, 8> args;
  DenseMap<BlockArgument, unsigned> argIndex;
  CommonPlan common;
  mutable DenseMap<Value, PointerExpr> pointerExprs;
  DenseSet<Value> canonicalInfrastructureValues;
  DenseSet<Value> requiredDataValues;
};

class FunctionLowerer {
public:
  FunctionLowerer(ModuleOp module, FunctionPlanner &planner)
      : module(module), planner(planner), ctx(module.getContext()), builder(ctx),
        typeAdapter(ctx) {}

  LogicalResult lower() {
    if (failed(createValueFunction()))
      return failure();
    if (failed(emitCommonPrefix()))
      return failure();

    for (Operation &op : planner.getEntryBlock()) {
      if (hasName(&op, kTTReturnOpName)) {
        builder.create<func::ReturnOp>(op.getLoc());
        continue;
      }
      if (planner.isCanonicalInfrastructure(&op))
        continue;
      if (failed(lowerOperation(&op)))
        return failure();
    }

    if (valueFunc.getBody().front().empty() ||
        !valueFunc.getBody().front().back().hasTrait<OpTrait::IsTerminator>())
      builder.create<func::ReturnOp>(planner.getFuncOp()->getLoc());

    // Unlink the parsed Triton function after emitting its value-layer
    // replacement. Destroying this generic Triton function during the pass trips
    // MLIR region teardown assertions on internal block-argument uses with the
    // pinned Triton/LLVM build; unlinking is enough to keep the output module at
    // the value-layer boundary.
    planner.getFuncOp()->remove();
    return success();
  }

private:
  LogicalResult createValueFunction() {
    Operation *sourceFunc = planner.getFuncOp();
    auto fn = llvm::dyn_cast<FunctionOpInterface>(sourceFunc);
    if (!fn)
      return sourceFunc->emitOpError("expected tt.func to implement FunctionOpInterface");

    SmallVector<Type, 8> argTypes;
    for (const TTIRArgInfo &arg : planner.getArgs()) {
      if (arg.isPointer) {
        Type elemType = typeAdapter.getValueElementType(arg.pointerElement, builder);
        if (!elemType)
          return sourceFunc->emitOpError("unsupported pointer element type");
        auto memorySpace = mlir::vc4value::GlobalMemorySpaceAttr::get(ctx);
        argTypes.push_back(MemRefType::get(
            {ShapedType::kDynamic}, elemType, MemRefLayoutAttrInterface(),
            memorySpace));
        continue;
      }
      Type converted = convertScalarArgType(arg.sourceType, arg.isSizeLike, builder);
      if (!converted)
        return sourceFunc->emitOpError("unsupported scalar argument type");
      argTypes.push_back(converted);
    }

    std::string name = sanitizeSymbolName(fn.getName());
    builder.setInsertionPoint(sourceFunc);
    valueFunc = builder.create<func::FuncOp>(
        sourceFunc->getLoc(), name, FunctionType::get(ctx, argTypes, {}));
    valueFunc->setAttr(kVC4ValueKernelAttr, builder.getUnitAttr());
    valueFunc->setAttr(kVC4ValueGridRankAttr, builder.getI32IntegerAttr(1));
    if (functionNeedsFiniteFPDomain())
      valueFunc->setAttr(kVC4ValueFPDomainAttr, builder.getStringAttr("finite"));

    Block *entry = valueFunc.addEntryBlock();
    for (auto [index, arg] : llvm::enumerate(planner.getArgs())) {
      valueFunc.setArgAttr(index, kVC4ValueArgNameAttr,
                           builder.getStringAttr(arg.valueName));
      if (arg.isPointer) {
        StringRef direction = "in";
        if (arg.loaded && arg.stored)
          direction = "inout";
        else if (arg.stored)
          direction = "out";
        valueFunc.setArgAttr(index, kVC4ValueDirectionAttr,
                             builder.getStringAttr(direction));
        if (planner.getCommonPlan().sizeArg) {
          valueFunc.setArgAttr(index, kVC4ValueShapeArgsAttr,
                               builder.getArrayAttr({builder.getStringAttr("n")}));
        }
      }
      valueMap[arg.sourceArg] = entry->getArgument(index);
      if (arg.isPointer) {
        PointerExpr ptr;
        ptr.sourcePointerArg = arg.sourceArg;
        ptr.valueMemref = entry->getArgument(index);
        ptr.element = arg.pointerElement;
        pointerValues[arg.sourceArg] = ptr;
      }
    }

    builder.setInsertionPointToStart(entry);
    return success();
  }

  bool functionNeedsFiniteFPDomain() const {
    bool needs = false;
    planner.getEntryBlock().walk([&](arith::CmpFOp cmp) {
      (void)cmp;
      needs = true;
    });
    return needs;
  }

  LogicalResult emitCommonPrefix() {
    const CommonPlan &common = planner.getCommonPlan();
    Value n = lookup(common.sizeArg);
    if (!n)
      return emitInternalError(planner.getFuncOp(), "tail size argument not mapped");

    Location loc = common.programId ? common.programId->getLoc()
                                    : planner.getFuncOp()->getLoc();
    Value pid = createVC4ValueProgramId(builder, loc, 0);
    (void)createVectorStep(builder, loc);
    Value c16 = builder.create<arith::ConstantIndexOp>(loc, 16);
    baseIndex = builder.create<arith::MulIOp>(loc, pid, c16);
    Value remaining = builder.create<arith::SubIOp>(loc, n, baseIndex);
    tailMask = createVectorCreateMask(builder, loc, remaining);

    valueMap[common.programId->getResult(0)] = pid;
    valueMap[common.offsetValue] = baseIndex; // Represents the scalar transfer base.
    valueMap[common.tailMaskValue] = tailMask;
    return success();
  }

  Value lookup(Value value) const {
    auto it = valueMap.find(value);
    if (it == valueMap.end())
      return {};
    return it->second;
  }

  LogicalResult lowerOperation(Operation *op) {
    if (isForbiddenProducerOrBackendDialect(op))
      return emitPermanentReject(op, "backend/non-TTIR dialect operation");

    if (op->getNumResults() > 0) {
      bool anyRequired = false;
      for (Value result : op->getResults())
        anyRequired |= planner.isRequiredDataValue(result);
      if (!anyRequired && !hasName(op, kTTLoadOpName))
        return success();
    }

    if (hasName(op, kTTSplatOpName))
      return lowerTTSplat(op);
    if (hasName(op, kTTAddPtrOpName))
      return lowerTTAddPtr(op);
    if (hasName(op, kTTLoadOpName))
      return lowerTTLoad(op);
    if (hasName(op, kTTStoreOpName))
      return lowerTTStore(op);

    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "arith")
      return lowerArith(op);

    // Pure TTIR arithmetic proof ops that were not part of the canonical value
    // dataflow can be ignored if all their results have no users that require a
    // lowered value.  They are range/overflow guards emitted by Triton before
    // the canonical offset.  If a later op asks for their value, lookup() will
    // fail there with a precise diagnostic.
    if (hasName(op, "arith.extsi") || hasName(op, "arith.andi") ||
        hasName(op, "arith.constant"))
      return success();

    if (hasName(op, kTTGetNumProgramsOpName))
      return emitStagedDiagnostic(op, "tt.get_num_programs");
    if (hasName(op, kTTDotOpName))
      return emitStagedDiagnostic(op, "tt.dot / contract");
    if (hasName(op, kTTReduceOpName) || hasName(op, kTTReduceReturnOpName))
      return emitStagedDiagnostic(op, "tt.reduce");
    if (hasName(op, kTTAdvanceOpName))
      return emitStagedDiagnostic(op, "block-pointer tt.advance");

    if (isTritonDialectOp(op))
      return emitStagedDiagnostic(op, op->getName().getStringRef());

    return emitStagedDiagnostic(op, op->getName().getStringRef());
  }

  LogicalResult lowerTTSplat(Operation *op) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return emitStagedDiagnostic(op, "non-unary tt.splat");

    Value src = op->getOperand(0);
    if (auto blockArg = llvm::dyn_cast<BlockArgument>(src)) {
      auto sourceArg = lookupSourceArg(blockArg);
      if (sourceArg && sourceArg->isPointer) {
        PointerExpr ptr = pointerValues[blockArg];
        pointerValues[op->getResult(0)] = ptr;
        return success();
      }
    }

    Type resultType =
        typeAdapter.convertTensorToValueVector(op->getResult(0).getType(), builder);
    if (!resultType)
      return emitStagedDiagnostic(op, "tt.splat result type");
    Value scalar = lookup(src);
    if (!scalar)
      return emitStagedDiagnostic(op, "tt.splat source not available in value IR");
    valueMap[op->getResult(0)] = createVectorBroadcast(builder, op->getLoc(), scalar, resultType);
    return success();
  }

  LogicalResult lowerTTAddPtr(Operation *op) {
    FailureOr<PointerExpr> ptr = planner.classifyPointer(op->getResult(0));
    if (failed(ptr))
      return failure();
    PointerExpr lowered = *ptr;
    lowered.valueMemref = pointerValues[ptr->sourcePointerArg].valueMemref;
    pointerValues[op->getResult(0)] = lowered;
    return success();
  }

  LogicalResult lowerTTLoad(Operation *op) {
    if (op->getNumResults() == 1 && !planner.isRequiredDataValue(op->getResult(0)))
      return success();
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return emitStagedDiagnostic(op, "tt.load without pointer, mask, other=0");
    if (op->getOperand(1) != planner.getCommonPlan().tailMaskValue)
      return emitStagedDiagnostic(op, "tt.load mask other than canonical tail mask");
    Operation *other = op->getOperand(2).getDefiningOp();
    if (!isZeroLikeConstant(other))
      return emitStagedDiagnostic(op, "tt.load other value other than zero");

    FailureOr<PointerExpr> ptr = planner.classifyPointer(op->getOperand(0));
    if (failed(ptr))
      return failure();
    if (!ptr->hasCanonicalOffset)
      return emitStagedDiagnostic(op, "tt.load pointer without canonical addptr offset");

    Type resultType =
        typeAdapter.convertTensorToValueVector(op->getResult(0).getType(), builder);
    auto vectorType = llvm::dyn_cast_or_null<VectorType>(resultType);
    if (!vectorType)
      return emitStagedDiagnostic(op, "tt.load result type");
    Value memref = pointerValues[ptr->sourcePointerArg].valueMemref;
    Value pad = createZeroPadding(op->getLoc(), vectorType.getElementType());
    valueMap[op->getResult(0)] =
        createVectorTransferRead(builder, op->getLoc(), memref, baseIndex, pad,
                                 tailMask, vectorType);
    return success();
  }

  LogicalResult lowerTTStore(Operation *op) {
    if (op->getNumOperands() != 3)
      return emitStagedDiagnostic(op, "tt.store without pointer, value, mask");
    if (op->getOperand(2) != planner.getCommonPlan().tailMaskValue)
      return emitStagedDiagnostic(op, "tt.store mask other than canonical tail mask");

    FailureOr<PointerExpr> ptr = planner.classifyPointer(op->getOperand(0));
    if (failed(ptr))
      return failure();
    if (!ptr->hasCanonicalOffset)
      return emitStagedDiagnostic(op, "tt.store pointer without canonical addptr offset");
    Value value = lookup(op->getOperand(1));
    if (!value)
      return emitStagedDiagnostic(op, "tt.store value was not produced by supported compute DAG");
    Value memref = pointerValues[ptr->sourcePointerArg].valueMemref;
    createVectorTransferWrite(builder, op->getLoc(), value, memref, baseIndex, tailMask);
    return success();
  }

  LogicalResult lowerArith(Operation *op) {
    if (auto constant = llvm::dyn_cast<arith::ConstantOp>(op))
      return lowerArithConstant(constant);
    if (hasName(op, "arith.extsi"))
      return success();
    if (hasName(op, "arith.andi"))
      return success();
    if (hasName(op, "arith.cmpi") && op->getResult(0) == planner.getCommonPlan().tailMaskValue)
      return success();

    static constexpr StringRef supportedArithOps[] = {
        "arith.addf", "arith.subf", "arith.mulf", "arith.addi",
        "arith.subi", "arith.cmpf", "arith.cmpi", "arith.select"};
    if (!hasAnyName(op, supportedArithOps))
      return emitStagedDiagnostic(op, op->getName().getStringRef());

    SmallVector<Value, 4> operands;
    operands.reserve(op->getNumOperands());
    for (Value operand : op->getOperands()) {
      Value mapped = lookup(operand);
      if (!mapped)
        return emitStagedDiagnostic(op, "arith operand not available in supported value dataflow");
      operands.push_back(mapped);
    }

    SmallVector<Type, 2> resultTypes;
    for (Type type : op->getResultTypes()) {
      Type mapped = convertResultType(type);
      if (!mapped)
        return emitStagedDiagnostic(op, "arith result type");
      resultTypes.push_back(mapped);
    }

    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    for (NamedAttribute attr : op->getAttrs())
      state.addAttribute(attr.getName(), attr.getValue());
    Operation *created = builder.create(state);
    for (auto [oldResult, newResult] : llvm::zip(op->getResults(), created->getResults()))
      valueMap[oldResult] = newResult;
    return success();
  }

  LogicalResult lowerArithConstant(arith::ConstantOp op) {
    if (op->getNumResults() != 1)
      return emitStagedDiagnostic(op, "multi-result arith.constant");
    Type resultType = convertResultType(op.getType());
    if (!resultType)
      return success(); // Ignore constants used only by canonical proof ops.

    Attribute attr = op.getValue();
    if (auto dense = llvm::dyn_cast<DenseElementsAttr>(attr)) {
      auto shaped = llvm::dyn_cast<ShapedType>(resultType);
      if (!shaped)
        return emitStagedDiagnostic(op, "dense constant without shaped result");
      FailureOr<Attribute> retargeted = retargetDenseAttr(dense, shaped);
      if (failed(retargeted))
        return emitStagedDiagnostic(op, "dense constant element type");
      attr = *retargeted;
    }
    auto typedAttr = llvm::dyn_cast<TypedAttr>(attr);
    if (!typedAttr)
      return emitStagedDiagnostic(op, "arith.constant without typed attr");
    valueMap[op.getResult()] =
        builder.create<arith::ConstantOp>(op.getLoc(), resultType, typedAttr);
    return success();
  }

  Type convertResultType(Type type) {
    if (type.isIndex() || type.isInteger(1) || type.isSignlessInteger(32) ||
        type.isF32())
      return type;
    if (auto vector = typeAdapter.convertTensorToValueVector(type, builder))
      return vector;
    return {};
  }

  Value createZeroPadding(Location loc, Type elementType) {
    if (elementType.isF32())
      return builder.create<arith::ConstantOp>(loc, elementType,
                                               builder.getFloatAttr(elementType, 0.0));
    if (elementType.isSignlessInteger(32))
      return builder.create<arith::ConstantOp>(loc, elementType,
                                               builder.getIntegerAttr(elementType, 0));
    return {};
  }

  TTIRArgInfo *lookupSourceArg(BlockArgument arg) { return planner.lookupArg(arg); }

  ModuleOp module;
  FunctionPlanner &planner;
  MLIRContext *ctx;
  OpBuilder builder;
  TTIRTypeAdapter typeAdapter;
  func::FuncOp valueFunc;
  DenseMap<Value, Value> valueMap;
  DenseMap<Value, PointerExpr> pointerValues;
  Value baseIndex;
  Value tailMask;
};

struct ConvertTritonToVC4ValuePass
    : public PassWrapper<ConvertTritonToVC4ValuePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTritonToVC4ValuePass)

  StringRef getArgument() const final { return "convert-triton-to-vc4-value"; }

  StringRef getDescription() const final {
    return "Lower the Phase 7.5 real TTIR elementwise subset to the VC4 value surface";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                    func::FuncDialect, math::MathDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect, mlir::vc4value::VC4ValueDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<Operation *, 4> ttFuncs;
    SmallVector<Operation *, 4> illegalTopLevelOps;

    for (Operation &op : module.getBody()->getOperations()) {
      if (hasName(&op, kTTFuncOpName)) {
        ttFuncs.push_back(&op);
        continue;
      }
      // Non-operation location aliases are not module body ops.  Any real
      // top-level op beside tt.func would survive into value IR and must be
      // classified explicitly by a later phase.
      illegalTopLevelOps.push_back(&op);
    }

    if (ttFuncs.empty()) {
      module.emitError("expected at least one tt.func; C++ TTIR-to-VC4Value "
                       "lowering remains Phase 7.5 elementwise V1 only. "
                       "READY_FOR_TRITON remains NO");
      signalPassFailure();
      return;
    }
    if (!illegalTopLevelOps.empty()) {
      illegalTopLevelOps.front()->emitOpError()
          << "top-level operation beside tt.func is not Phase 7.5 lowerable; "
          << "READY_FOR_TRITON remains NO";
      signalPassFailure();
      return;
    }

    for (Operation *ttFunc : llvm::make_early_inc_range(ttFuncs)) {
      FunctionPlanner planner(ttFunc);
      if (failed(planner.analyze())) {
        signalPassFailure();
        return;
      }
      FunctionLowerer lowerer(module, planner);
      if (failed(lowerer.lower())) {
        signalPassFailure();
        return;
      }
    }

    // Defensive boundary check: the importer must produce only value-layer IR.
    bool sawIllegal = false;
    module.walk([&](Operation *op) {
      if (op == module.getOperation())
        return WalkResult::advance();
      StringRef dialect = op->getName().getDialectNamespace();
      if (dialect == "tt" || dialect == "ttg" || dialect == "triton_gpu" ||
          dialect == "nvgpu" || dialect == "nvvm" || dialect == "gpu" ||
          dialect == "vc4kernel" || dialect == "ssavc4" || dialect == "vc4") {
        op->emitOpError() << "TTIR importer emitted or preserved forbidden dialect '"
                          << dialect << "'";
        sawIllegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (sawIllegal)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vc4::createConvertTritonToVC4ValuePass() {
  return std::make_unique<ConvertTritonToVC4ValuePass>();
}

void mlir::vc4::registerConvertTritonToVC4ValuePass() {
  // Keep the explicit registration API used by the rest of the VC4 conversion
  // stack.  The file-scope PassRegistration below installs the pass when this
  // translation unit is linked into the optional Triton frontend tool.
}

static PassRegistration<ConvertTritonToVC4ValuePass>
    registerConvertTritonToVC4ValuePass;
