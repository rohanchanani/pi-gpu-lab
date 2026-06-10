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
//   * Unsupported TTIR forms are classified as staged Phase 8.5 limitations
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
#include "mlir/IR/OwningOpRef.h"
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
#include <set>
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
constexpr llvm::StringLiteral kTTMakeBlockPtrOpName("tt.make_block_ptr");
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

static LogicalResult emitStagedDiagnostic(Operation *op, const Twine &detail) {
  return op->emitOpError()
         << detail << " is not Phase 8.5 C++ TTIR-to-VC4Value lowerable; "
         << "staged TTIR target-profile feature. READY_FOR_TRITON remains NO";
}

static LogicalResult emitStagedBodyFeatureDiagnostic(Operation *op,
                                                     const Twine &detail) {
  return op->emitOpError()
         << detail
         << " staged by body feature, not unsupported control flow; "
         << "READY_FOR_TRITON remains NO";
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

enum class LoweringOutcome {
  LoweredWithResultsBound,
  LoweredZeroResult,
  IgnoredDeadProofOp,
  StagedUnsupported,
  InternalError
};

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

static bool isForbiddenValueOutputDialect(StringRef dialect) {
  return dialect == "tt" || dialect == "ttg" || dialect == "triton_gpu" ||
         dialect == "nvgpu" || dialect == "nvvm" || dialect == "gpu" ||
         dialect == "llvm" || dialect == "spirv" || dialect == "vc4kernel" ||
         dialect == "ssavc4" || dialect == "vc4";
}

static bool isAllowedValueOutputDialect(StringRef dialect) {
  return dialect == "builtin" || dialect == "func" || dialect == "vc4value" ||
         dialect == "vector" || dialect == "memref" || dialect == "arith" ||
         dialect == "math" || dialect == "scf" || dialect == "cf";
}

static bool isObservedOptimizationOnlyLoopAttr(NamedAttribute attr) {
  StringRef name = attr.getName().strref();
  return name == "tt.loop_unroll_factor" || name == "tt.flatten";
}

static LogicalResult copyValueSafeAttrs(Operation *source,
                                        OperationState &state) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef dialect = attr.getName().strref().split('.').first;
    if (dialect == "tt") {
      if (isObservedOptimizationOnlyLoopAttr(attr))
        continue;
      return emitStagedDiagnostic(source, Twine("unsupported TTIR attribute ") +
                                              attr.getName().strref());
    }
    state.addAttribute(attr.getName(), attr.getValue());
  }
  return success();
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

static bool isOrContainsI64(Type type) {
  if (type.isSignlessInteger(64))
    return true;
  if (auto shaped = llvm::dyn_cast<ShapedType>(type))
    return shaped.getElementType().isSignlessInteger(64);
  return false;
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

enum class TTIRArgumentRole {
  GlobalPointerIn,
  GlobalPointerOut,
  GlobalPointerInOut,
  ScalarTailBound,
  ScalarF32,
  ScalarI32,
  Unknown
};

static Type convertScalarArgType(Type type, TTIRArgumentRole role,
                                 OpBuilder &builder) {
  if (role == TTIRArgumentRole::ScalarTailBound && isScalarI32(type))
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
    std::string cleaned = sanitizeSymbolName(name);
    if (!cleaned.empty())
      return cleaned;
  }
  return ("arg" + Twine(index)).str();
}

static std::string makeUniqueMetadataName(StringRef base,
                                          std::set<std::string> &usedNames) {
  std::string sanitized = sanitizeSymbolName(base);
  if (usedNames.insert(sanitized).second)
    return sanitized;
  for (unsigned suffix = 1;; ++suffix) {
    std::string candidate = (Twine(sanitized) + "_" + Twine(suffix)).str();
    if (usedNames.insert(candidate).second)
      return candidate;
  }
}

static bool hasAnyName(Operation *op, ArrayRef<StringRef> names) {
  StringRef name = op->getName().getStringRef();
  return llvm::is_contained(names, name);
}

static bool isDeadProofWhitelistOp(Operation *op) {
  if (!op || op->getNumRegions() != 0 || op->getNumSuccessors() != 0 ||
      op->getNumResults() == 0)
    return false;
  if (hasName(op, "arith.extsi") || hasName(op, "arith.andi") ||
      hasName(op, "arith.trunci") || hasName(op, "arith.index_cast"))
    return true;
  if (hasName(op, "ub.poison"))
    return true;
  if (hasName(op, "arith.constant")) {
    for (Type type : op->getResultTypes())
      if (isOrContainsI64(type))
        return true;
    return false;
  }
  if (hasName(op, "arith.addi") || hasName(op, "arith.subi") ||
      hasName(op, "arith.muli") || hasName(op, "arith.cmpi")) {
    for (Type type : op->getOperandTypes())
      if (isOrContainsI64(type))
        return true;
    for (Type type : op->getResultTypes())
      if (isOrContainsI64(type))
        return true;
  }
  return false;
}

enum class TTIRProgramAxis { X = 0, Y = 1, Z = 2 };

struct TTIRRangeInfo {
  int64_t start = 0;
  int64_t end = 0;
  int64_t width = 0;
};

class TTIRAttrAdapter {
public:
  FailureOr<TTIRProgramAxis> classifyProgramAxis(Operation *op) const {
    if (auto programId = llvm::dyn_cast<mlir::triton::GetProgramIdOp>(op))
      return convertProgramAxis(op, programId.getAxis());
    if (auto numPrograms = llvm::dyn_cast<mlir::triton::GetNumProgramsOp>(op))
      return convertProgramAxis(op, numPrograms.getAxis());

    auto axisAttr = op->getAttrOfType<mlir::triton::ProgramIDDimAttr>("axis");
    if (!axisAttr)
      return op->emitOpError("unknown program axis attr form");
    return convertProgramAxis(op, axisAttr.getValue());
  }

  FailureOr<TTIRRangeInfo> classifyMakeRange(Operation *op) const {
    int64_t start = 0;
    int64_t end = 0;
    if (auto makeRange = llvm::dyn_cast<mlir::triton::MakeRangeOp>(op)) {
      start = static_cast<int64_t>(makeRange.getStart());
      end = static_cast<int64_t>(makeRange.getEnd());
    } else {
      auto startAttr = op->getAttrOfType<IntegerAttr>("start");
      auto endAttr = op->getAttrOfType<IntegerAttr>("end");
      if (!startAttr || !endAttr)
        return op->emitOpError("unknown tt.make_range start/end attr form");
      start = startAttr.getInt();
      end = endAttr.getInt();
    }

    TTIRRangeInfo info;
    info.start = start;
    info.end = end;
    info.width = end - start;
    return info;
  }

private:
  FailureOr<TTIRProgramAxis>
  convertProgramAxis(Operation *op, mlir::triton::ProgramIDDim axis) const {
    switch (axis) {
    case mlir::triton::ProgramIDDim::X:
      return TTIRProgramAxis::X;
    case mlir::triton::ProgramIDDim::Y:
      return TTIRProgramAxis::Y;
    case mlir::triton::ProgramIDDim::Z:
      return TTIRProgramAxis::Z;
    }
    return op->emitOpError("unknown program axis enum value");
  }
};

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
  TTIRArgumentRole role = TTIRArgumentRole::Unknown;
  bool isPointer = false;
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
      return emitStagedDiagnostic(funcOp, "tt.func without a body region");

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
    classifyScalarArgumentRoles();
    return success();
  }

  Operation *getFuncOp() const { return funcOp; }
  Block &getEntryBlock() const { return funcOp->getRegion(0).front(); }
  ArrayRef<TTIRArgInfo> getArgs() const { return args; }
  const CommonPlan &getCommonPlan() const { return common; }
  StringRef getTailBoundArgName() const {
    auto it = argIndex.find(common.sizeArg);
    if (it == argIndex.end())
      return {};
    return args[it->second].valueName;
  }

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
  bool hasRequiredResult(Operation *op) const {
    if (!op)
      return false;
    for (Value result : op->getResults())
      if (requiredDataValues.contains(result))
        return true;
    return false;
  }

  bool isCanonicalInfrastructure(Operation *op) const {
    if (op == common.programId || op == common.makeRange)
      return true;
    if (op->getNumResults() == 0)
      return false;
    for (Value result : op->getResults()) {
      if (requiredDataValues.contains(result))
        return false;
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

  bool canIgnoreAsDeadProofOp(Operation *op) const {
    DenseSet<Operation *> visiting;
    DenseMap<Operation *, bool> memo;
    return canIgnoreAsDeadProofOp(op, visiting, memo);
  }

private:
  LogicalResult collectArgs(Block &entry) {
    args.reserve(entry.getNumArguments());
    std::set<std::string> usedArgNames;
    for (auto [index, arg] : llvm::enumerate(entry.getArguments())) {
      TTIRArgInfo info;
      info.sourceArg = arg;
      info.sourceType = arg.getType();
      info.valueName =
          makeUniqueMetadataName(inferSourceArgName(arg, index), usedArgNames);
      if (auto pointer = typeAdapter.classifyPointerType(arg.getType())) {
        if (pointer->isTensorPointer)
          return emitStagedDiagnostic(funcOp, "block pointer or tensor pointer argument");
        if (!pointer->isSupportedPhase75Element)
          return emitStagedDiagnostic(funcOp, "unsupported pointer argument element type");
        info.isPointer = true;
        info.pointerElement = pointer->pointeeKind;
      } else if (isScalarI32(arg.getType()) || isScalarF32(arg.getType())) {
        info.role = isScalarF32(arg.getType()) ? TTIRArgumentRole::ScalarF32
                                               : TTIRArgumentRole::ScalarI32;
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

  bool canIgnoreAsDeadProofOp(Operation *op, DenseSet<Operation *> &visiting,
                              DenseMap<Operation *, bool> &memo) const {
    if (!isDeadProofWhitelistOp(op))
      return false;
    if (hasRequiredResult(op))
      return false;
    if (auto found = memo.find(op); found != memo.end())
      return found->second;
    if (!visiting.insert(op).second)
      return false;

    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!canIgnoreAsDeadProofOp(user, visiting, memo)) {
          visiting.erase(op);
          memo[op] = false;
          return false;
        }
      }
    }

    visiting.erase(op);
    memo[op] = true;
    return true;
  }

  LogicalResult scanOps(Block &entry) {
    (void)entry;
    // Prefer high-level staged feature diagnostics before more generic tensor
    // shape diagnostics.  This keeps future dot/reduction tests from being
    // masked by their rank-2 operands or helper constants.
    Operation *stagedBodyFeature = nullptr;
    funcOp->walk([&](Operation *op) {
      if (stagedBodyFeature)
        return WalkResult::interrupt();
      if (hasName(op, kTTDotOpName) || hasName(op, kTTReduceOpName) ||
          hasName(op, kTTReduceReturnOpName) ||
          hasName(op, kTTMakeBlockPtrOpName) ||
          hasName(op, kTTAdvanceOpName)) {
        stagedBodyFeature = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (stagedBodyFeature) {
      if (hasName(stagedBodyFeature, kTTDotOpName))
        return emitStagedBodyFeatureDiagnostic(stagedBodyFeature,
                                               "tt.dot / contract");
      if (hasName(stagedBodyFeature, kTTReduceOpName) ||
          hasName(stagedBodyFeature, kTTReduceReturnOpName))
        return emitStagedBodyFeatureDiagnostic(stagedBodyFeature,
                                               "tt.reduce");
      return emitStagedBodyFeatureDiagnostic(stagedBodyFeature,
                                             "block-pointer TTIR op");
    }

    bool sawFailure = false;
    funcOp->walk([&](Operation *op) {
      if (op == funcOp || sawFailure)
        return WalkResult::advance();
      if (isForbiddenProducerOrBackendDialect(op)) {
        (void)emitPermanentReject(
            op, "backend/non-TTIR dialect operation in TTIR input");
        sawFailure = true;
        return WalkResult::interrupt();
      }
      if (hasName(op, kTTGetNumProgramsOpName)) {
        FailureOr<TTIRProgramAxis> axis = attrAdapter.classifyProgramAxis(op);
        if (failed(axis)) {
          sawFailure = true;
          return WalkResult::interrupt();
        }
        if (*axis != TTIRProgramAxis::X) {
          (void)emitStagedDiagnostic(
              op,
              "num_programs axis 1/2 is staged until value grid-rank >1 lowering");
          sawFailure = true;
          return WalkResult::interrupt();
        }
      }
      if (hasName(op, kTTBroadcastOpName) || hasName(op, kTTExpandDimsOpName) ||
          hasName(op, kTTTransOpName)) {
        (void)emitStagedDiagnostic(op,
                                   "rank-2/shape-changing TTIR tensor form");
        sawFailure = true;
        return WalkResult::interrupt();
      }
      for (Type type : op->getResultTypes()) {
        if (containsRankGreaterThanOneTensor(type)) {
          (void)emitStagedDiagnostic(op,
                                     "rank-2 or higher TTIR tensor result");
          sawFailure = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (sawFailure)
      return failure();
    return success();
  }

  LogicalResult classifyCommonPlan() {
    SmallVector<Operation *, 2> programIds;
    SmallVector<Operation *, 2> ranges;
    SmallVector<Operation *, 2> stores;
    SmallVector<Operation *, 4> addptrs;
    funcOp->walk([&](Operation *op) {
      if (hasName(op, kTTGetProgramIdOpName))
        programIds.push_back(op);
      if (hasName(op, kTTMakeRangeOpName))
        ranges.push_back(op);
      if (hasName(op, kTTStoreOpName))
        stores.push_back(op);
      if (hasName(op, kTTAddPtrOpName))
        addptrs.push_back(op);
    });

    if (programIds.size() != 1)
      return emitStagedDiagnostic(funcOp, "TTIR functions with other than one program_id");
    FailureOr<TTIRProgramAxis> axis =
        attrAdapter.classifyProgramAxis(programIds.front());
    if (failed(axis))
      return failure();
    if (*axis != TTIRProgramAxis::X)
      return emitStagedDiagnostic(
          programIds.front(),
          "program_id axis 1/2 is staged until value grid-rank >1 lowering");
    common.programId = programIds.front();

    if (ranges.size() != 1)
      return emitStagedDiagnostic(funcOp, "tt.make_range other than 0..16");
    FailureOr<TTIRRangeInfo> range = attrAdapter.classifyMakeRange(ranges.front());
    if (failed(range))
      return failure();
    if (range->start != 0 || range->end != 16 || range->width != 16)
      return emitStagedDiagnostic(ranges.front(), "tt.make_range other than 0..16");
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

    Value scalarBase = mul->getOperand(0) == common.programId->getResult(0)
                           ? mul->getOperand(0)
                           : mul->getOperand(1) == common.programId->getResult(0)
                                 ? mul->getOperand(1)
                                 : Value();
    if (!scalarBase) {
      if (llvm::isa<BlockArgument>(mul->getOperand(0)))
        scalarBase = mul->getOperand(0);
      else if (llvm::isa<BlockArgument>(mul->getOperand(1)))
        scalarBase = mul->getOperand(1);
      else
        return emitStagedDiagnostic(
            mul, "pointer offset base missing program_id or scalar loop iv");
    }

    Value factor = mul->getOperand(0) == scalarBase
                       ? mul->getOperand(1)
                       : mul->getOperand(0);
    if (auto cst = factor.getDefiningOp<arith::ConstantOp>()) {
      if (auto integer = llvm::dyn_cast<IntegerAttr>(cst.getValue())) {
        if (integer.getInt() == 16) {
          canonicalInfrastructureValues.insert(factor);
          canonicalInfrastructureValues.insert(other);
          canonicalInfrastructureValues.insert(offset);
          markRequiredValue(splat->getOperand(0));
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
    if (it == argIndex.end())
      return emitInternalError(rhsSplat, "tail mask size argument missing from arg table");
    if (!isScalarI32(args[it->second].sourceType))
      return emitStagedDiagnostic(rhsSplat, "tail mask bound is not an i32 scalar argument");
    args[it->second].role = TTIRArgumentRole::ScalarTailBound;
    common.sizeArg = sizeArg;
    canonicalInfrastructureValues.insert(rhsSplat->getResult(0));
    canonicalInfrastructureValues.insert(mask);
    return success();
  }

  LogicalResult classifyMemoryDirections() {
    bool sawFailure = false;
    funcOp->walk([&](Operation *op) {
      if (sawFailure)
        return WalkResult::advance();
      if (!hasName(op, kTTLoadOpName) && !hasName(op, kTTStoreOpName))
        return WalkResult::advance();
      if (op->getNumOperands() < 1) {
        (void)emitStagedDiagnostic(op, "memory op without pointer operand");
        sawFailure = true;
        return WalkResult::interrupt();
      }
      FailureOr<PointerExpr> ptr = classifyPointer(op->getOperand(0));
      if (failed(ptr)) {
        sawFailure = true;
        return WalkResult::interrupt();
      }
      auto it = argIndex.find(ptr->sourcePointerArg);
      if (it == argIndex.end()) {
        (void)emitInternalError(op, "pointer base argument missing from arg table");
        sawFailure = true;
        return WalkResult::interrupt();
      }
      if (hasName(op, kTTLoadOpName))
        args[it->second].loaded = true;
      if (hasName(op, kTTStoreOpName))
        args[it->second].stored = true;
      return WalkResult::advance();
    });
    if (sawFailure)
      return failure();
    for (TTIRArgInfo &arg : args) {
      if (!arg.isPointer)
        continue;
      if (arg.loaded && arg.stored)
        arg.role = TTIRArgumentRole::GlobalPointerInOut;
      else if (arg.stored)
        arg.role = TTIRArgumentRole::GlobalPointerOut;
      else if (arg.loaded)
        arg.role = TTIRArgumentRole::GlobalPointerIn;
    }
    return success();
  }

  void classifyScalarArgumentRoles() {
    for (TTIRArgInfo &arg : args) {
      if (arg.isPointer || arg.sourceArg == common.sizeArg)
        continue;
      if (isScalarF32(arg.sourceType)) {
        arg.role = TTIRArgumentRole::ScalarF32;
        continue;
      }
      if (isScalarI32(arg.sourceType))
        arg.role = TTIRArgumentRole::ScalarI32;
    }
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

    if (def->getNumRegions() > 0) {
      std::optional<unsigned> resultIndex;
      if (auto result = llvm::dyn_cast<OpResult>(value))
        resultIndex = result.getResultNumber();
      def->walk([&](Operation *nested) {
        if (!nested->hasTrait<OpTrait::IsTerminator>())
          return;
        if (hasName(nested, "scf.yield") && resultIndex &&
            *resultIndex < nested->getNumOperands()) {
          markRequiredValue(nested->getOperand(*resultIndex));
          return;
        }
        if (hasName(nested, "scf.condition")) {
          for (Value operand : nested->getOperands())
            markRequiredValue(operand);
          return;
        }
        for (Value operand : nested->getOperands())
          markRequiredValue(operand);
      });
    }

    for (Value operand : def->getOperands())
      markRequiredValue(operand);
  }

  void markRequiredDataflow() {
    funcOp->walk([&](Operation *op) {
      if (hasName(op, kTTStoreOpName) && op->getNumOperands() >= 2)
        markRequiredValue(op->getOperand(1));
      if (op->getName().getDialectNamespace() == "scf" ||
          op->getName().getDialectNamespace() == "cf") {
        for (Value operand : op->getOperands())
          markRequiredValue(operand);
      }
    });
  }

  Operation *funcOp;
  TTIRTypeAdapter typeAdapter;
  TTIRAttrAdapter attrAdapter;
  SmallVector<TTIRArgInfo, 8> args;
  DenseMap<BlockArgument, unsigned> argIndex;
  CommonPlan common;
  mutable DenseMap<Value, PointerExpr> pointerExprs;
  DenseSet<Value> canonicalInfrastructureValues;
  DenseSet<Value> requiredDataValues;
};

class FunctionLowerer {
public:
  FunctionLowerer(ModuleOp outputModule, FunctionPlanner &planner)
      : outputModule(outputModule), planner(planner),
        ctx(outputModule.getContext()), builder(ctx), typeAdapter(ctx) {}

  LogicalResult lower() {
    if (failed(createValueFunction()))
      return failure();
    if (failed(emitCommonPrefix()))
      return failure();

    if (failed(lowerBlock(planner.getEntryBlock(), valueFunc.getBody().front())))
      return failure();

    if (valueFunc.getBody().front().empty() ||
        !valueFunc.getBody().front().back().hasTrait<OpTrait::IsTerminator>())
      builder.create<func::ReturnOp>(planner.getFuncOp()->getLoc());

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
      Type converted = convertScalarArgType(arg.sourceType, arg.role, builder);
      if (!converted)
        return sourceFunc->emitOpError("unsupported scalar argument type");
      argTypes.push_back(converted);
    }

    std::string name = sanitizeSymbolName(fn.getName());
    builder.setInsertionPointToEnd(outputModule.getBody());
    valueFunc = builder.create<func::FuncOp>(
        sourceFunc->getLoc(), name, FunctionType::get(ctx, argTypes, {}));
    valueFunc->setAttr(kVC4ValueKernelAttr, builder.getUnitAttr());
    valueFunc->setAttr(kVC4ValueGridRankAttr, builder.getI32IntegerAttr(1));
    if (functionNeedsFiniteFPDomain())
      valueFunc->setAttr(kVC4ValueFPDomainAttr, builder.getStringAttr("finite"));

    Block *entry = valueFunc.addEntryBlock();
    for (auto [index, arg] : llvm::enumerate(planner.getArgs())) {
      // Argument names are metadata only. Semantic roles are derived from TTIR
      // SSA/use-def structure, not source names.
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
                               builder.getArrayAttr(
                                   {builder.getStringAttr(
                                       planner.getTailBoundArgName())}));
        }
      }
      bindValue(arg.sourceArg, entry->getArgument(index));
      if (arg.isPointer) {
        PointerExpr ptr;
        ptr.sourcePointerArg = arg.sourceArg;
        ptr.valueMemref = entry->getArgument(index);
        ptr.element = arg.pointerElement;
        bindPointer(arg.sourceArg, ptr);
      }
    }

    blockMap[&planner.getEntryBlock()] = entry;
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
    Value pidIndex = createVC4ValueProgramId(builder, loc, 0);
    Value pidI32 =
        builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), pidIndex);

    bindValue(common.programId->getResult(0), pidI32);
    return success();
  }

  Value lookup(Value value) const {
    auto it = valueMap.find(value);
    if (it == valueMap.end())
      return {};
    return it->second;
  }

  void bindValue(Value source, Value mapped) {
    if (!source || !mapped)
      return;
    if (!valueScopes.empty()) {
      auto found = valueMap.find(source);
      valueScopes.back().push_back(
          {source, found == valueMap.end() ? Value() : found->second});
    }
    valueMap[source] = mapped;
  }

  void bindPointer(Value source, const PointerExpr &mapped) {
    if (!source)
      return;
    if (!pointerScopes.empty()) {
      auto found = pointerValues.find(source);
      pointerScopes.back().push_back({source, found == pointerValues.end()
                                                  ? std::optional<PointerExpr>()
                                                  : std::optional<PointerExpr>(
                                                        found->second)});
    }
    pointerValues[source] = mapped;
  }

  void pushValueScope() {
    valueScopes.emplace_back();
    pointerScopes.emplace_back();
  }

  void popValueScope() {
    for (auto it = valueScopes.back().rbegin(), e = valueScopes.back().rend();
         it != e; ++it) {
      if (it->oldValue)
        valueMap[it->source] = it->oldValue;
      else
        valueMap.erase(it->source);
    }
    valueScopes.pop_back();

    for (auto it = pointerScopes.back().rbegin(), e = pointerScopes.back().rend();
         it != e; ++it) {
      if (it->oldValue)
        pointerValues[it->source] = *it->oldValue;
      else
        pointerValues.erase(it->source);
    }
    pointerScopes.pop_back();
  }

  bool isResultAccounted(Value result) const {
    return valueMap.contains(result) || pointerValues.contains(result) ||
           planner.isCanonicalInfrastructure(result.getDefiningOp());
  }

  LogicalResult finishLowering(Operation *op, LoweringOutcome outcome) {
    loweringOutcomes[op] = outcome;
    switch (outcome) {
    case LoweringOutcome::LoweredWithResultsBound:
      for (Value result : op->getResults()) {
        if (!isResultAccounted(result))
          return emitInternalError(
              op, Twine("result of ") + op->getName().getStringRef() +
                      " was not mapped by successful lowering");
      }
      return success();
    case LoweringOutcome::LoweredZeroResult:
      if (op->getNumResults() != 0)
        return emitInternalError(
            op, Twine("result-producing ") + op->getName().getStringRef() +
                    " reported zero-result lowering");
      return success();
    case LoweringOutcome::IgnoredDeadProofOp:
      if (!planner.canIgnoreAsDeadProofOp(op))
        return emitInternalError(
            op, Twine("ignored-dead proof failed for ") +
                    op->getName().getStringRef());
      return success();
    case LoweringOutcome::StagedUnsupported:
      return emitStagedDiagnostic(op, op->getName().getStringRef());
    case LoweringOutcome::InternalError:
      return emitInternalError(op, op->getName().getStringRef());
    }
    return emitInternalError(op, "unknown lowering outcome");
  }

  LogicalResult lowerOperation(Operation *op) {
    if (isForbiddenProducerOrBackendDialect(op))
      return emitPermanentReject(op, "backend/non-TTIR dialect operation");

    if (hasName(op, kTTReturnOpName)) {
      builder.create<func::ReturnOp>(op->getLoc());
      return finishLowering(op, LoweringOutcome::LoweredZeroResult);
    }

    if (op == planner.getCommonPlan().programId)
      return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
    if (op == planner.getCommonPlan().makeRange) {
      if (!planner.isRequiredDataValue(op->getResult(0)))
        return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
      return lowerTTMakeRange(op);
    }

    if (planner.isCanonicalInfrastructure(op))
      return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);

    if (isDeadProofWhitelistOp(op)) {
      if (planner.hasRequiredResult(op)) {
        if (hasName(op, "arith.extsi"))
          return emitStagedBodyFeatureDiagnostic(
              op, "arith.extsi required use staged");
        if (hasName(op, "arith.andi"))
          return emitStagedBodyFeatureDiagnostic(
              op, "arith.andi required use staged");
      }
      if (planner.canIgnoreAsDeadProofOp(op))
        return finishLowering(op, LoweringOutcome::IgnoredDeadProofOp);
      if (hasName(op, "arith.extsi"))
        return emitStagedBodyFeatureDiagnostic(
            op, "arith.extsi required use staged");
      if (hasName(op, "arith.andi"))
        return emitStagedBodyFeatureDiagnostic(
            op, "arith.andi required use staged");
    }

    if (hasName(op, kTTSplatOpName))
      return lowerTTSplat(op);
    if (hasName(op, kTTMakeRangeOpName))
      return lowerTTMakeRange(op);
    if (hasName(op, kTTAddPtrOpName))
      return lowerTTAddPtr(op);
    if (hasName(op, kTTLoadOpName))
      return lowerTTLoad(op);
    if (hasName(op, kTTStoreOpName))
      return lowerTTStore(op);
    if (hasName(op, kTTGetNumProgramsOpName))
      return lowerTTGetNumPrograms(op);

    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "arith")
      return lowerArith(op);
    if (dialect == "scf")
      return lowerSCF(op);
    if (dialect == "cf")
      return lowerCF(op);

    if (hasName(op, kTTDotOpName))
      return emitStagedBodyFeatureDiagnostic(op, "tt.dot / contract");
    if (hasName(op, kTTReduceOpName) || hasName(op, kTTReduceReturnOpName))
      return emitStagedBodyFeatureDiagnostic(op, "tt.reduce");
    if (hasName(op, kTTMakeBlockPtrOpName))
      return emitStagedBodyFeatureDiagnostic(op, "tt.make_block_ptr");
    if (hasName(op, kTTAdvanceOpName))
      return emitStagedBodyFeatureDiagnostic(op, "block-pointer tt.advance");

    if (isTritonDialectOp(op))
      return emitStagedDiagnostic(op, op->getName().getStringRef());

    return emitStagedDiagnostic(op, op->getName().getStringRef());
  }

  LogicalResult lowerTTMakeRange(Operation *op) {
    if (op->getNumResults() != 1)
      return emitStagedDiagnostic(op, "tt.make_range result shape");
    FailureOr<TTIRRangeInfo> range = TTIRAttrAdapter().classifyMakeRange(op);
    if (failed(range))
      return failure();
    if (range->start != 0 || range->end != kPhase7VectorWidth)
      return emitStagedDiagnostic(op, "tt.make_range outside 0..16");

    Type resultType = convertResultType(op->getResult(0).getType());
    auto vectorType = llvm::dyn_cast_or_null<VectorType>(resultType);
    if (!vectorType || vectorType.getRank() != 1 ||
        vectorType.getDimSize(0) != kPhase7VectorWidth ||
        !vectorType.getElementType().isSignlessInteger(32))
      return emitStagedDiagnostic(op, "tt.make_range result type");

    SmallVector<APInt, kPhase7VectorWidth> values;
    values.reserve(kPhase7VectorWidth);
    for (int64_t lane = 0; lane < kPhase7VectorWidth; ++lane)
      values.push_back(APInt(32, lane));
    auto attr = DenseIntElementsAttr::get(vectorType, values);
    bindValue(op->getResult(0),
              builder.create<arith::ConstantOp>(op->getLoc(), vectorType, attr));
    return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
  }

  LogicalResult lowerTTGetNumPrograms(Operation *op) {
    if (op->getNumResults() != 1)
      return emitStagedDiagnostic(op, "tt.get_num_programs result shape");
    FailureOr<TTIRProgramAxis> axis = TTIRAttrAdapter().classifyProgramAxis(op);
    if (failed(axis))
      return failure();
    if (*axis != TTIRProgramAxis::X)
      return emitStagedDiagnostic(
          op, "num_programs axis 1/2 is staged until value grid-rank >1 lowering");
    Value numProgramsIndex = createGenericOpWithResult(
        builder, op->getLoc(), kVC4ValueNumProgramsOpName, {},
        {builder.getNamedAttr("axis", builder.getI32IntegerAttr(0))},
        builder.getIndexType());
    bindValue(op->getResult(0), builder.create<arith::IndexCastOp>(
                                      op->getLoc(), builder.getI32Type(),
                                      numProgramsIndex));
    return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
  }

  LogicalResult lowerTTSplat(Operation *op) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return emitStagedDiagnostic(op, "non-unary tt.splat");

    Value src = op->getOperand(0);
    if (auto blockArg = llvm::dyn_cast<BlockArgument>(src)) {
      auto sourceArg = lookupSourceArg(blockArg);
      if (sourceArg && sourceArg->isPointer) {
        PointerExpr ptr = pointerValues[blockArg];
        bindPointer(op->getResult(0), ptr);
        return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
      }
    }

    Type resultType =
        typeAdapter.convertTensorToValueVector(op->getResult(0).getType(), builder);
    if (!resultType)
      return emitStagedDiagnostic(op, "tt.splat result type");
    Value scalar = lookup(src);
    if (!scalar)
      return emitStagedDiagnostic(op, "tt.splat source not available in value IR");
    bindValue(op->getResult(0),
              createVectorBroadcast(builder, op->getLoc(), scalar, resultType));
    return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
  }

  LogicalResult lowerTTAddPtr(Operation *op) {
    FailureOr<PointerExpr> ptr = planner.classifyPointer(op->getResult(0));
    if (failed(ptr))
      return failure();
    PointerExpr lowered = *ptr;
    lowered.valueMemref = pointerValues[ptr->sourcePointerArg].valueMemref;
    bindPointer(op->getResult(0), lowered);
    return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
  }

  LogicalResult lowerTTLoad(Operation *op) {
    if (op->getNumResults() == 1 && !planner.isRequiredDataValue(op->getResult(0)))
      return emitStagedDiagnostic(op, "dead tt.load outside ignored-dead proof whitelist");
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return emitStagedDiagnostic(op, "tt.load without pointer, mask, other=0");
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
    FailureOr<Value> transferIndex = ensureTransferIndex(ptr->offsetValue, op);
    if (failed(transferIndex))
      return failure();
    FailureOr<Value> mask = ensureTailMask(op->getOperand(1), op);
    if (failed(mask))
      return failure();
    bindValue(op->getResult(0),
              createVectorTransferRead(builder, op->getLoc(), memref,
                                       *transferIndex, pad, *mask, vectorType));
    return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
  }

  LogicalResult lowerTTStore(Operation *op) {
    if (op->getNumOperands() != 3)
      return emitStagedDiagnostic(op, "tt.store without pointer, value, mask");
    FailureOr<PointerExpr> ptr = planner.classifyPointer(op->getOperand(0));
    if (failed(ptr))
      return failure();
    if (!ptr->hasCanonicalOffset)
      return emitStagedDiagnostic(op, "tt.store pointer without canonical addptr offset");
    Value value = lookup(op->getOperand(1));
    if (!value)
      return emitStagedDiagnostic(op, "tt.store value was not produced by supported compute DAG");
    Value memref = pointerValues[ptr->sourcePointerArg].valueMemref;
    FailureOr<Value> transferIndex = ensureTransferIndex(ptr->offsetValue, op);
    if (failed(transferIndex))
      return failure();
    FailureOr<Value> mask = ensureTailMask(op->getOperand(2), op);
    if (failed(mask))
      return failure();
    createVectorTransferWrite(builder, op->getLoc(), value, memref,
                              *transferIndex, *mask);
    return finishLowering(op, LoweringOutcome::LoweredZeroResult);
  }

  LogicalResult lowerArith(Operation *op) {
    if (auto constant = llvm::dyn_cast<arith::ConstantOp>(op))
      return lowerArithConstant(constant);
    if (hasName(op, "arith.sitofp"))
      return emitStagedBodyFeatureDiagnostic(op, "arith.sitofp");
    if (hasName(op, "arith.bitcast")) {
      if (op->getNumOperands() != 1 || op->getNumResults() != 1)
        return emitStagedDiagnostic(op, "non-unary arith.bitcast");
      Value mapped = lookup(op->getOperand(0));
      if (!mapped)
        return emitStagedDiagnostic(op, "arith.bitcast operand not available");
      Type resultType = convertResultType(op->getResult(0).getType());
      if (!resultType || resultType != mapped.getType())
        return emitStagedDiagnostic(op, "arith.bitcast changing value type");
      bindValue(op->getResult(0), mapped);
      return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
    }
    if (hasName(op, "arith.extsi"))
      return emitStagedBodyFeatureDiagnostic(op,
                                             "arith.extsi required use staged");
    if (hasName(op, "arith.andi"))
      return emitStagedBodyFeatureDiagnostic(op,
                                             "arith.andi required use staged");
    if (hasName(op, "arith.cmpi") &&
        op->getResult(0) == planner.getCommonPlan().tailMaskValue) {
      FailureOr<Value> mask = ensureTailMask(op->getResult(0), op);
      return failed(mask) ? failure()
                          : finishLowering(
                                op, LoweringOutcome::LoweredWithResultsBound);
    }

    static constexpr StringRef supportedArithOps[] = {
        "arith.addf", "arith.subf", "arith.mulf", "arith.addi",
        "arith.subi", "arith.muli", "arith.cmpf", "arith.cmpi",
        "arith.select"};
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
      bindValue(oldResult, newResult);
    return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
  }

  LogicalResult lowerArithConstant(arith::ConstantOp op) {
    if (op->getNumResults() != 1)
      return emitStagedDiagnostic(op, "multi-result arith.constant");
    Type resultType = convertResultType(op.getType());
    if (!resultType)
      return emitStagedDiagnostic(op, "arith.constant result type");

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
    bindValue(op.getResult(),
              builder.create<arith::ConstantOp>(op.getLoc(), resultType,
                                                typedAttr));
    return finishLowering(op.getOperation(),
                          LoweringOutcome::LoweredWithResultsBound);
  }

  Type convertResultType(Type type) {
    if (type.isIndex() || type.isInteger(1) || type.isSignlessInteger(32) ||
        type.isF32())
      return type;
    if (auto vector = typeAdapter.convertTensorToValueVector(type, builder))
      return vector;
    return {};
  }

  Type convertRegionValueType(Type type) { return convertResultType(type); }

  LogicalResult lowerBlock(Block &sourceBlock, Block &destBlock) {
    builder.setInsertionPointToEnd(&destBlock);
    for (Operation &op : sourceBlock) {
      if (failed(lowerOperation(&op)))
        return failure();
    }
    return success();
  }

  LogicalResult lowerRegion(Region &source, Region &dest) {
    if (!dest.empty())
      return emitInternalError(planner.getFuncOp(),
                               "destination region unexpectedly non-empty");

    SmallVector<std::pair<Block *, Block *>, 4> blockPairs;
    for (Block &sourceBlock : source) {
      auto *destBlock = new Block();
      dest.push_back(destBlock);
      blockMap[&sourceBlock] = destBlock;
      for (BlockArgument arg : sourceBlock.getArguments()) {
        Type converted = convertRegionValueType(arg.getType());
        if (!converted)
          return emitStagedDiagnostic(source.getParentOp(),
                                      "unsupported region block argument type");
        destBlock->addArgument(converted, arg.getLoc());
      }
      blockPairs.push_back({&sourceBlock, destBlock});
    }

    pushValueScope();
    for (auto [sourceBlock, destBlock] : blockPairs) {
      for (auto [oldArg, newArg] :
           llvm::zip(sourceBlock->getArguments(), destBlock->getArguments()))
        bindValue(oldArg, newArg);
    }

    for (auto [sourceBlock, destBlock] : blockPairs) {
      if (failed(lowerBlock(*sourceBlock, *destBlock))) {
        popValueScope();
        return failure();
      }
    }
    popValueScope();
    return success();
  }

  LogicalResult lowerSCF(Operation *op) {
    if (hasName(op, "scf.yield"))
      return lowerGenericTerminator(op);
    if (hasName(op, "scf.condition")) {
      if (op->getNumOperands() < 1 || !op->getOperand(0).getType().isInteger(1))
        return emitPermanentReject(op, "vector/per-lane branch condition as CFG");
      return lowerGenericTerminator(op);
    }
    if (!hasAnyName(op, {"scf.if", "scf.for", "scf.while"}))
      return emitStagedDiagnostic(op, op->getName().getStringRef());

    if (hasName(op, "scf.if") &&
        (op->getNumOperands() != 1 ||
         !op->getOperand(0).getType().isInteger(1)))
      return emitPermanentReject(op, "vector/per-lane branch condition as CFG");

    SmallVector<Value, 8> operands;
    for (Value operand : op->getOperands()) {
      Value mapped = lookup(operand);
      if (!mapped)
        return emitStagedDiagnostic(op, "SCF operand not available in value map");
      operands.push_back(mapped);
    }

    SmallVector<Type, 4> resultTypes;
    for (Type type : op->getResultTypes()) {
      Type converted = convertRegionValueType(type);
      if (!converted)
        return emitStagedDiagnostic(op, "SCF result type");
      resultTypes.push_back(converted);
    }

    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    if (failed(copyValueSafeAttrs(op, state)))
      return failure();
    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      state.addRegion();
    Operation *created = builder.create(state);
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), created->getResults()))
      bindValue(oldResult, newResult);

    for (auto [sourceRegion, destRegion] :
         llvm::zip(op->getRegions(), created->getRegions()))
      if (failed(lowerRegion(sourceRegion, destRegion)))
        return failure();
    builder.setInsertionPointAfter(created);
    return finishLowering(op, op->getNumResults() == 0
                                  ? LoweringOutcome::LoweredZeroResult
                                  : LoweringOutcome::LoweredWithResultsBound);
  }

  LogicalResult lowerGenericTerminator(Operation *op) {
    SmallVector<Value, 8> operands;
    for (Value operand : op->getOperands()) {
      Value mapped = lookup(operand);
      if (!mapped)
        return emitStagedDiagnostic(op, "terminator operand not available");
      operands.push_back(mapped);
    }
    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands(operands);
    if (failed(copyValueSafeAttrs(op, state)))
      return failure();
    builder.create(state);
    return finishLowering(op, LoweringOutcome::LoweredZeroResult);
  }

  LogicalResult lowerCF(Operation *op) {
    if (hasName(op, "cf.switch"))
      return emitStagedDiagnostic(op,
                                  "cf.switch was not emitted by Phase85a2 TTIR");
    if (!hasAnyName(op, {"cf.br", "cf.cond_br"}))
      return emitStagedDiagnostic(op, op->getName().getStringRef());
    if (hasName(op, "cf.cond_br") &&
        (op->getNumOperands() < 1 ||
         !op->getOperand(0).getType().isInteger(1)))
      return emitPermanentReject(op, "vector/per-lane branch condition as CFG");

    SmallVector<Value, 8> operands;
    for (Value operand : op->getOperands()) {
      Value mapped = lookup(operand);
      if (!mapped)
        return emitStagedDiagnostic(op, "CF operand not available in value map");
      operands.push_back(mapped);
    }

    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands(operands);
    for (Block *successor : op->getSuccessors()) {
      auto it = blockMap.find(successor);
      if (it == blockMap.end())
        return emitInternalError(op, "CF successor block missing from block map");
      state.addSuccessors(it->second);
    }
    if (failed(copyValueSafeAttrs(op, state)))
      return failure();
    builder.create(state);
    return finishLowering(op, LoweringOutcome::LoweredZeroResult);
  }

  FailureOr<Value> ensureTransferIndex(Value offset, Operation *user) {
    Operation *add = offset.getDefiningOp();
    if (!hasName(add, "arith.addi") || add->getNumOperands() != 2)
      return emitStagedDiagnostic(user, "pointer offset not formed by arith.addi");

    Value rangeResult = planner.getCommonPlan().makeRange->getResult(0);
    Value other;
    if (add->getOperand(0) == rangeResult)
      other = add->getOperand(1);
    else if (add->getOperand(1) == rangeResult)
      other = add->getOperand(0);
    else
      return emitStagedDiagnostic(user, "pointer offset missing tt.make_range lanes");

    Operation *splat = other.getDefiningOp();
    if (!hasName(splat, kTTSplatOpName) || splat->getNumOperands() != 1)
      return emitStagedDiagnostic(user,
                                  "pointer offset base not formed by tt.splat");

    Value scalarBase = lookup(splat->getOperand(0));
    if (!scalarBase)
      return emitStagedDiagnostic(user, "pointer scalar base not available");
    if (scalarBase.getType().isIndex())
      return scalarBase;
    if (!scalarBase.getType().isSignlessInteger(32))
      return emitStagedDiagnostic(user, "pointer scalar base is not i32/index");
    return builder.create<arith::IndexCastOp>(user->getLoc(),
                                              builder.getIndexType(),
                                              scalarBase)
        .getResult();
  }

  FailureOr<Value> ensureTailMask(Value mask, Operation *user) {
    if (mask != planner.getCommonPlan().tailMaskValue)
      return emitStagedDiagnostic(user, "non-canonical control-flow tail mask");

    Operation *cmp = mask.getDefiningOp();
    auto cmpi = llvm::dyn_cast_or_null<arith::CmpIOp>(cmp);
    if (!cmpi)
      return emitStagedDiagnostic(user, "tail mask not formed by arith.cmpi");
    FailureOr<Value> index = ensureTransferIndex(cmpi.getOperand(0), user);
    if (failed(index))
      return failure();
    Value n = lookup(planner.getCommonPlan().sizeArg);
    if (!n)
      return emitInternalError(user, "tail size argument not mapped");
    Value remaining = builder.create<arith::SubIOp>(user->getLoc(), n, *index);
    return createVectorCreateMask(builder, user->getLoc(), remaining);
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

  struct ScopedValueBinding {
    Value source;
    Value oldValue;
  };

  struct ScopedPointerBinding {
    Value source;
    std::optional<PointerExpr> oldValue;
  };

  ModuleOp outputModule;
  FunctionPlanner &planner;
  MLIRContext *ctx;
  OpBuilder builder;
  TTIRTypeAdapter typeAdapter;
  func::FuncOp valueFunc;
  DenseMap<Value, Value> valueMap;
  SmallVector<SmallVector<ScopedValueBinding, 16>, 4> valueScopes;
  SmallVector<SmallVector<ScopedPointerBinding, 16>, 4> pointerScopes;
  DenseMap<Block *, Block *> blockMap;
  DenseMap<Value, PointerExpr> pointerValues;
  DenseMap<Operation *, LoweringOutcome> loweringOutcomes;
};

class TTIRToValueModuleBuilder {
public:
  explicit TTIRToValueModuleBuilder(ModuleOp outputModule)
      : outputModule(outputModule) {}

  LogicalResult lower(ArrayRef<std::unique_ptr<FunctionPlanner>> planners) {
    for (const std::unique_ptr<FunctionPlanner> &planner : planners) {
      FunctionLowerer lowerer(outputModule, *planner);
      if (failed(lowerer.lower()))
        return failure();
    }
    return success();
  }

private:
  ModuleOp outputModule;
};

static LogicalResult validateInputModuleAttrs(ModuleOp inputModule) {
  if (inputModule->getAttrs().empty())
    return success();

  InFlightDiagnostic diag = inputModule.emitError()
      << "TTIR module attributes are not Phase 7.5 C++ TTIR-to-VC4Value "
         "lowerable; unknown module attrs are staged until a value-safe module "
         "attribute policy is implemented. READY_FOR_TRITON remains NO";
  for (NamedAttribute attr : inputModule->getAttrs())
    diag << "\n  attr: " << attr.getName();
  return failure();
}

static LogicalResult verifyValueOutputModule(ModuleOp outputModule) {
  for (Operation &op : outputModule.getBody()->getOperations()) {
    if (!llvm::isa<func::FuncOp>(&op))
      return op.emitOpError()
             << "top-level value output operation must be func.func";

    if (!op.hasAttr(kVC4ValueKernelAttr))
      return op.emitOpError()
             << "value output function missing vc4value.kernel";
    if (!op.hasAttr(kVC4ValueGridRankAttr))
      return op.emitOpError()
             << "value output function missing vc4value.grid_rank";
  }

  bool sawIllegal = false;
  outputModule.walk([&](Operation *op) {
    StringRef dialect = op->getName().getDialectNamespace();
    if (isForbiddenValueOutputDialect(dialect)) {
      op->emitOpError() << "TTIR importer emitted or preserved forbidden dialect '"
                        << dialect << "'";
      sawIllegal = true;
      return WalkResult::interrupt();
    }
    if (!isAllowedValueOutputDialect(dialect)) {
      op->emitOpError() << "TTIR importer emitted unexpected value output dialect '"
                        << dialect << "'";
      sawIllegal = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (sawIllegal)
    return failure();
  return success();
}

static LogicalResult commitOutputModule(ModuleOp inputModule,
                                        ModuleOp outputModule) {
  if (failed(verifyValueOutputModule(outputModule)))
    return failure();

  Block *inputBody = inputModule.getBody();
  SmallVector<Operation *, 8> oldOps;
  for (Operation &op : inputBody->getOperations())
    oldOps.push_back(&op);

  for (Operation *op : oldOps)
    op->erase();

  Block *outputBody = outputModule.getBody();
  inputBody->getOperations().splice(inputBody->end(),
                                    outputBody->getOperations());

  return verifyValueOutputModule(inputModule);
}

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
    ModuleOp inputModule = getOperation();
    SmallVector<Operation *, 4> ttFuncs;
    SmallVector<Operation *, 4> illegalTopLevelOps;

    if (failed(validateInputModuleAttrs(inputModule))) {
      signalPassFailure();
      return;
    }

    for (Operation &op : inputModule.getBody()->getOperations()) {
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
      inputModule.emitError("expected at least one tt.func; C++ TTIR-to-VC4Value "
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

    OwningOpRef<ModuleOp> outputModule =
        ModuleOp::create(inputModule.getLoc());

    {
      SmallVector<std::unique_ptr<FunctionPlanner>, 4> planners;
      std::set<std::string> valueFunctionNames;

      for (Operation *ttFunc : ttFuncs) {
        auto planner = std::make_unique<FunctionPlanner>(ttFunc);
        if (failed(planner->analyze())) {
          signalPassFailure();
          return;
        }

        auto fn = llvm::dyn_cast<FunctionOpInterface>(ttFunc);
        if (!fn) {
          ttFunc->emitOpError("expected tt.func to implement FunctionOpInterface");
          signalPassFailure();
          return;
        }
        std::string valueName = sanitizeSymbolName(fn.getName());
        if (!valueFunctionNames.insert(valueName).second) {
          ttFunc->emitOpError()
              << "duplicate value output symbol '" << valueName
              << "' after TTIR function name sanitization is not Phase 7.5 "
                 "lowerable; READY_FOR_TRITON remains NO";
          signalPassFailure();
          return;
        }

        planners.push_back(std::move(planner));
      }

      TTIRToValueModuleBuilder moduleBuilder(*outputModule);
      if (failed(moduleBuilder.lower(planners))) {
        signalPassFailure();
        return;
      }
    }

    if (failed(verifyValueOutputModule(*outputModule))) {
      signalPassFailure();
      return;
    }

    if (failed(commitOutputModule(inputModule, *outputModule)))
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
