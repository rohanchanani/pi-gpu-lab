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
//   * Unsupported TTIR forms are staged against the current VC4 TTIR target
//     profile unless the profile has a permanent reject proof.
//
// The importer started as the Phase 7 elementwise V1 subset and now grows
// through the locked vertical feature ladder. It still emits only standard
// value-layer IR; later layers lower value IR to VC4Kernel and below.
//
//   tt.get_program_id axis 0
//   tt.make_range 0..16
//   contiguous base + pid*16 + arange pointer expressions
//   tail mask offsets < n
//   masked tt.load with other=0
//   masked tt.store with the same tail mask
//   straight-line i32/f32 add/sub/mul/cmp/select compute DAGs
//
// Future phases should extend the semantic planners below
// (launch/range/pointer/mask/memory/compute/reduction/math/dot) instead of
// adding kernel-name templates.
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
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
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
constexpr llvm::StringLiteral kTTCallOpName("tt.call");
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
constexpr llvm::StringLiteral
    kVC4ValueReductionPolicyAttr("vc4value.reduction_policy");
constexpr llvm::StringLiteral kVC4ValueMaxPolicyAttr("vc4value.max_policy");
constexpr llvm::StringLiteral
    kVC4ValueF16StoragePolicyAttr("vc4value.f16_storage_policy");
constexpr llvm::StringLiteral kVC4ValueMathPolicyAttr("vc4value.math_policy");

constexpr llvm::StringLiteral kVC4ValueProgramIdOpName("vc4value.program_id");
constexpr llvm::StringLiteral
    kVC4ValueNumProgramsOpName("vc4value.num_programs");

constexpr unsigned kPhase7VectorWidth = 16;

static bool hasName(Operation *op, StringRef name) {
  return op && op->getName().getStringRef() == name;
}

static LogicalResult emitStagedDiagnostic(Operation *op, const Twine &detail) {
  return op->emitOpError()
         << detail
         << " is not currently lowerable by the VC4 TTIR target profile; "
         << "staged TTIR target-profile feature. READY_FOR_TRITON remains NO";
}

static LogicalResult emitStagedBodyFeatureDiagnostic(Operation *op,
                                                     const Twine &detail) {
  return op->emitOpError()
         << detail << " staged by body feature, not unsupported control flow; "
         << "READY_FOR_TRITON remains NO";
}

static LogicalResult emitPermanentReject(Operation *op, const Twine &detail) {
  return op->emitOpError() << detail
                           << " is rejected by the VC4 TTIR target profile. "
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
         dialect == "nvgpu" || dialect == "nvvm" || dialect == "rocdl" ||
         dialect == "gpu" || dialect == "llvm" || dialect == "spirv" ||
         dialect == "tensor" || dialect == "linalg" || dialect == "stablehlo" ||
         dialect == "mhlo" || dialect == "tosa" || dialect == "iree" ||
         dialect == "vc4kernel" || dialect == "ssavc4" || dialect == "vc4";
}

static bool isAllowedValueOutputBodyDialect(StringRef dialect) {
  return dialect == "vc4value" || dialect == "vector" || dialect == "memref" ||
         dialect == "arith" || dialect == "math" || dialect == "scf" ||
         dialect == "cf";
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
static bool isScalarF16(Type type) { return type.isF16(); }
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

static bool isBF16OrFP8Type(Type type) {
  if (type.isBF16())
    return true;
  auto floatType = llvm::dyn_cast<FloatType>(type);
  return floatType && floatType.getWidth() == 8;
}

static Type getStorageScalarElement(Type type) {
  if (auto shaped = llvm::dyn_cast<RankedTensorType>(type))
    return shaped.getElementType();
  return type;
}

static LogicalResult emitUnsupportedPointerElementDiagnostic(Operation *op,
                                                            Type type) {
  Type elementType = getStorageScalarElement(type);
  if (isBF16OrFP8Type(elementType))
    return op->emitOpError()
           << "bf16/fp8 storage is staged; READY_FOR_TRITON remains NO";
  if (elementType.isSignlessInteger(8) || elementType.isSignlessInteger(16))
    return op->emitOpError()
           << "int8/int16 quantized storage is staged; "
           << "READY_FOR_TRITON remains NO";
  return emitStagedDiagnostic(op, "unsupported pointer argument element type");
}

static bool isRankedTensorPointer(Type type) {
  auto shaped = llvm::dyn_cast<RankedTensorType>(type);
  return shaped &&
         llvm::isa<mlir::triton::PointerType>(shaped.getElementType());
}

static bool isScalarPointer(Type type) {
  return llvm::isa<mlir::triton::PointerType>(type);
}

static bool isPublicTTFunc(Operation *op) {
  if (!hasName(op, kTTFuncOpName))
    return false;
  auto visibility = op->getAttrOfType<StringAttr>("sym_visibility");
  return !visibility || visibility.getValue() == "public";
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

enum class TTIRReduceClassification {
  NotReduceCall,
  AddI32,
  AddF32,
  MaxF32,
  NonAdd,
  RankGreaterThanOne,
  MultiResult,
  Unsupported
};

struct TTIRReduceCallInfo {
  TTIRReduceClassification classification =
      TTIRReduceClassification::NotReduceCall;
  Operation *reduceOp = nullptr;
};

static Operation *lookupTTCallee(Operation *callOp) {
  if (!hasName(callOp, kTTCallOpName))
    return nullptr;
  auto callee = callOp->getAttrOfType<SymbolRefAttr>("callee");
  if (!callee)
    return nullptr;
  return SymbolTable::lookupNearestSymbolFrom(callOp, callee);
}

static Operation *getFirstReachableReturn(Operation *funcOp) {
  if (!funcOp || funcOp->getNumRegions() != 1 || funcOp->getRegion(0).empty())
    return nullptr;
  Block &entry = funcOp->getRegion(0).front();
  if (entry.empty())
    return nullptr;
  Operation &terminator = entry.back();
  return hasName(&terminator, kTTReturnOpName) ? &terminator : nullptr;
}

static bool valueIsOneOfBlockArgs(Value value, Block &block) {
  for (BlockArgument arg : block.getArguments())
    if (value == arg)
      return true;
  return false;
}

static bool operationAddsTwoBlockArgs(Operation *op, Type elementType,
                                      Block &block) {
  if (!op || op->getNumOperands() != 2 || op->getNumResults() != 1)
    return false;
  if (elementType.isF32()) {
    if (!hasName(op, "arith.addf"))
      return false;
  } else if (elementType.isSignlessInteger(32)) {
    if (!hasName(op, "arith.addi"))
      return false;
  } else {
    return false;
  }
  return valueIsOneOfBlockArgs(op->getOperand(0), block) &&
         valueIsOneOfBlockArgs(op->getOperand(1), block) &&
         op->getOperand(0) != op->getOperand(1);
}

static bool operationMaxesTwoBlockArgs(Operation *op, Type elementType,
                                       Block &block) {
  if (!op || !elementType.isF32() || op->getNumOperands() != 2 ||
      op->getNumResults() != 1)
    return false;
  if (!hasName(op, "arith.maxnumf") && !hasName(op, "arith.maximumf"))
    return false;
  return valueIsOneOfBlockArgs(op->getOperand(0), block) &&
         valueIsOneOfBlockArgs(op->getOperand(1), block) &&
         op->getOperand(0) != op->getOperand(1);
}

static bool functionReturnsAddOfArgs(Operation *funcOp, Type elementType) {
  Operation *ret = getFirstReachableReturn(funcOp);
  if (!ret || ret->getNumOperands() != 1 || funcOp->getRegion(0).empty())
    return false;
  Block &entry = funcOp->getRegion(0).front();
  if (entry.getNumArguments() != 2)
    return false;
  Operation *def = ret->getOperand(0).getDefiningOp();
  return operationAddsTwoBlockArgs(def, elementType, entry);
}

static bool functionReturnsMaxOfArgs(Operation *funcOp, Type elementType) {
  Operation *ret = getFirstReachableReturn(funcOp);
  if (!ret || !elementType.isF32() || ret->getNumOperands() != 1 ||
      funcOp->getRegion(0).empty())
    return false;
  Block &entry = funcOp->getRegion(0).front();
  if (entry.getNumArguments() != 2)
    return false;
  Operation *def = ret->getOperand(0).getDefiningOp();
  return operationMaxesTwoBlockArgs(def, elementType, entry);
}

static TTIRReduceClassification classifyReduceReturnValue(Value returned,
                                                          Block &reduceBlock,
                                                          Type elementType) {
  Operation *def = returned.getDefiningOp();
  if (operationAddsTwoBlockArgs(def, elementType, reduceBlock))
    return elementType.isF32() ? TTIRReduceClassification::AddF32
                               : TTIRReduceClassification::AddI32;
  if (operationMaxesTwoBlockArgs(def, elementType, reduceBlock))
    return TTIRReduceClassification::MaxF32;
  if (hasName(def, kTTCallOpName)) {
    if (def->getNumOperands() != 2 || def->getNumResults() != 1)
      return TTIRReduceClassification::Unsupported;
    if (!valueIsOneOfBlockArgs(def->getOperand(0), reduceBlock) ||
        !valueIsOneOfBlockArgs(def->getOperand(1), reduceBlock) ||
        def->getOperand(0) == def->getOperand(1))
      return TTIRReduceClassification::Unsupported;
    Operation *callee = lookupTTCallee(def);
    if (functionReturnsAddOfArgs(callee, elementType))
      return elementType.isF32() ? TTIRReduceClassification::AddF32
                                 : TTIRReduceClassification::AddI32;
    if (functionReturnsMaxOfArgs(callee, elementType))
      return TTIRReduceClassification::MaxF32;
    return TTIRReduceClassification::NonAdd;
  }
  return TTIRReduceClassification::NonAdd;
}

static TTIRReduceCallInfo classifyTTIRReduceCall(Operation *callOp) {
  TTIRReduceCallInfo info;
  Operation *callee = lookupTTCallee(callOp);
  if (!callee || !hasName(callee, kTTFuncOpName) ||
      callee->getNumRegions() != 1 || callee->getRegion(0).empty())
    return info;

  Operation *ret = getFirstReachableReturn(callee);
  if (!ret || ret->getNumOperands() != 1)
    return info;
  Operation *reduce = ret->getOperand(0).getDefiningOp();
  if (!hasName(reduce, kTTReduceOpName))
    return info;

  info.reduceOp = reduce;
  if (reduce->getNumOperands() != 1 || reduce->getNumResults() != 1 ||
      callOp->getNumResults() != 1) {
    info.classification = TTIRReduceClassification::MultiResult;
    return info;
  }

  auto axisAttr = reduce->getAttrOfType<IntegerAttr>("axis");
  if (!axisAttr) {
    info.classification = TTIRReduceClassification::Unsupported;
    return info;
  }
  if (axisAttr.getInt() != 0) {
    info.classification = TTIRReduceClassification::RankGreaterThanOne;
    return info;
  }

  auto inputTensor =
      llvm::dyn_cast<RankedTensorType>(reduce->getOperand(0).getType());
  if (!inputTensor) {
    info.classification = TTIRReduceClassification::Unsupported;
    return info;
  }
  if (inputTensor.getRank() > 1 ||
      containsRankGreaterThanOneTensor(callOp->getResult(0).getType())) {
    info.classification = TTIRReduceClassification::RankGreaterThanOne;
    return info;
  }
  if (inputTensor.getRank() != 1 || inputTensor.getDimSize(0) != 16) {
    info.classification = TTIRReduceClassification::Unsupported;
    return info;
  }

  Type elementType = inputTensor.getElementType();
  if (!elementType.isF32() && !elementType.isSignlessInteger(32)) {
    info.classification = TTIRReduceClassification::Unsupported;
    return info;
  }
  if (reduce->getResult(0).getType() != elementType ||
      callOp->getResult(0).getType() != elementType) {
    info.classification = TTIRReduceClassification::Unsupported;
    return info;
  }
  if (reduce->getNumRegions() != 1 || reduce->getRegion(0).empty()) {
    info.classification = TTIRReduceClassification::Unsupported;
    return info;
  }

  Block &combiner = reduce->getRegion(0).front();
  if (combiner.getNumArguments() != 2 ||
      combiner.getArgument(0).getType() != elementType ||
      combiner.getArgument(1).getType() != elementType || combiner.empty()) {
    info.classification = TTIRReduceClassification::Unsupported;
    return info;
  }
  Operation &terminator = combiner.back();
  if (!hasName(&terminator, kTTReduceReturnOpName) ||
      terminator.getNumOperands() != 1) {
    info.classification = TTIRReduceClassification::Unsupported;
    return info;
  }

  info.classification = classifyReduceReturnValue(terminator.getOperand(0),
                                                  combiner, elementType);
  return info;
}

static bool isTTIRAddF32ReduceCallValue(Value value) {
  Operation *def = value.getDefiningOp();
  if (!hasName(def, kTTCallOpName))
    return false;
  TTIRReduceCallInfo reduceInfo = classifyTTIRReduceCall(def);
  return reduceInfo.classification == TTIRReduceClassification::AddF32;
}

static bool isTTIRRegionBlockArgumentOf(Value value, Operation *parentOp) {
  auto arg = llvm::dyn_cast<BlockArgument>(value);
  return arg && arg.getOwner() && arg.getOwner()->getParentOp() == parentOp;
}

static bool isTTIRLoopCarriedF32ReduceAccumulation(Operation *op) {
  if (!hasName(op, "arith.addf") || op->getNumOperands() != 2 ||
      op->getNumResults() != 1 || !op->getResult(0).getType().isF32())
    return false;

  Operation *parent = op->getParentOp();
  while (parent && !hasName(parent, "scf.for"))
    parent = parent->getParentOp();
  if (!parent)
    return false;

  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  return (isTTIRRegionBlockArgumentOf(lhs, parent) &&
          isTTIRAddF32ReduceCallValue(rhs)) ||
         (isTTIRRegionBlockArgumentOf(rhs, parent) &&
          isTTIRAddF32ReduceCallValue(lhs));
}

static bool isTTIRLoadLikeVectorValue(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (hasName(def, kTTLoadOpName))
    return true;
  if ((hasName(def, "arith.extf") || hasName(def, "arith.truncf")) &&
      def->getNumOperands() == 1)
    return isTTIRLoadLikeVectorValue(def->getOperand(0));
  return false;
}

static bool valueTransitivelyFeedsMathExp(Value value,
                                          DenseSet<Value> &visiting) {
  if (!visiting.insert(value).second)
    return false;
  for (Operation *user : value.getUsers()) {
    if (llvm::isa<math::ExpOp>(user))
      return true;
    if (user->getNumResults() == 0)
      continue;
    if (hasName(user, "arith.addf") || hasName(user, "arith.subf") ||
        hasName(user, "arith.mulf") || hasName(user, "arith.divf") ||
        hasName(user, "arith.maxnumf") ||
        hasName(user, "arith.maximumf") || hasName(user, kTTSplatOpName)) {
      for (Value result : user->getResults())
        if (valueTransitivelyFeedsMathExp(result, visiting))
          return true;
    }
  }
  return false;
}

static bool isTTIRGeneratedScoreReductionCall(Operation *callOp) {
  if (!hasName(callOp, kTTCallOpName) || callOp->getNumResults() != 1)
    return false;
  TTIRReduceCallInfo reduceInfo = classifyTTIRReduceCall(callOp);
  if (reduceInfo.classification != TTIRReduceClassification::AddF32 ||
      callOp->getNumOperands() != 1)
    return false;
  Operation *product = callOp->getOperand(0).getDefiningOp();
  if (!hasName(product, "arith.mulf") || product->getNumOperands() != 2)
    return false;
  if (!isTTIRLoadLikeVectorValue(product->getOperand(0)) ||
      !isTTIRLoadLikeVectorValue(product->getOperand(1)))
    return false;
  DenseSet<Value> visiting;
  return valueTransitivelyFeedsMathExp(callOp->getResult(0), visiting);
}

enum class TTIRScalarElementKind { I1, I32, F16, F32, Unsupported };

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
        !info.isTensorPointer &&
        (info.pointeeKind == TTIRScalarElementKind::I32 ||
         info.pointeeKind == TTIRScalarElementKind::F16 ||
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

  Type getValueElementType(TTIRScalarElementKind kind,
                           OpBuilder &builder) const {
    switch (kind) {
    case TTIRScalarElementKind::I1:
      return builder.getI1Type();
    case TTIRScalarElementKind::I32:
      return builder.getI32Type();
    case TTIRScalarElementKind::F16:
      return builder.getF16Type();
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
    if (type.isF16())
      return TTIRScalarElementKind::F16;
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
  if (newType.getElementType().isF16()) {
    SmallVector<APFloat, 16> values;
    values.reserve(dense.getNumElements());
    for (APFloat value : dense.getValues<APFloat>()) {
      bool losesInfo = false;
      value.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven,
                    &losesInfo);
      values.push_back(value);
    }
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
  if (hasName(op, "arith.extf") || hasName(op, "arith.truncf"))
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
  if (op && (hasName(op, "arith.extf") || hasName(op, "arith.truncf") ||
             hasName(op, "arith.extsi") || hasName(op, "arith.trunci"))) {
    if (op->getNumOperands() != 1)
      return false;
    return isZeroLikeConstant(op->getOperand(0).getDefiningOp());
  }
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
  return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                        b.getAffineDimExpr(0));
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
  Operation *op =
      createGenericOp(builder, loc, name, operands, attrs, resultType);
  return op->getResult(0);
}

static Value createVC4ValueProgramId(OpBuilder &builder, Location loc,
                                     int64_t axis) {
  return createGenericOpWithResult(
      builder, loc, kVC4ValueProgramIdOpName, {},
      {builder.getNamedAttr("axis", builder.getI32IntegerAttr(axis))},
      builder.getIndexType());
}

static Value createVC4ValueNumPrograms(OpBuilder &builder, Location loc,
                                       int64_t axis) {
  return createGenericOpWithResult(
      builder, loc, kVC4ValueNumProgramsOpName, {},
      {builder.getNamedAttr("axis", builder.getI32IntegerAttr(axis))},
      builder.getIndexType());
}

static Value createVectorStep(OpBuilder &builder, Location loc) {
  return createGenericOpWithResult(
      builder, loc, "vector.step", {}, {},
      VectorType::get({16}, builder.getIndexType()));
}

static Value createVectorBroadcast(OpBuilder &builder, Location loc,
                                   Value value, Type resultType) {
  return createGenericOpWithResult(builder, loc, "vector.broadcast", value, {},
                                   resultType);
}

static Value createVectorCreateMask(OpBuilder &builder, Location loc,
                                    Value activeCount) {
  return createGenericOpWithResult(builder, loc, "vector.create_mask",
                                   activeCount, {},
                                   VectorType::get({16}, builder.getI1Type()));
}

static Value createVectorTransferRead(OpBuilder &builder, Location loc,
                                      Value source, Value index, Value padding,
                                      Value mask, VectorType resultType) {
  MLIRContext *ctx = builder.getContext();
  SmallVector<Value, 4> operands = {source, index, padding};
  if (mask)
    operands.push_back(mask);
  SmallVector<NamedAttribute, 4> attrs;
  attrs.push_back(builder.getNamedAttr(
      "permutation_map", AffineMapAttr::get(getRank1IdentityMap(ctx))));
  attrs.push_back(builder.getNamedAttr(
      "in_bounds", builder.getArrayAttr({builder.getBoolAttr(false)})));
  attrs.push_back(builder.getNamedAttr(
      "operandSegmentSizes",
      DenseI32ArrayAttr::get(ctx, {1, 1, 1, mask ? 1 : 0})));
  return createGenericOpWithResult(builder, loc, "vector.transfer_read",
                                   operands, attrs, resultType);
}

static void createVectorTransferWrite(OpBuilder &builder, Location loc,
                                      Value value, Value dest, Value index,
                                      Value mask) {
  MLIRContext *ctx = builder.getContext();
  SmallVector<Value, 4> operands = {value, dest, index};
  if (mask)
    operands.push_back(mask);
  SmallVector<NamedAttribute, 4> attrs;
  attrs.push_back(builder.getNamedAttr(
      "permutation_map", AffineMapAttr::get(getRank1IdentityMap(ctx))));
  attrs.push_back(builder.getNamedAttr(
      "in_bounds", builder.getArrayAttr({builder.getBoolAttr(false)})));
  attrs.push_back(builder.getNamedAttr(
      "operandSegmentSizes",
      DenseI32ArrayAttr::get(ctx, {1, 1, 1, mask ? 1 : 0})));
  createGenericOp(builder, loc, "vector.transfer_write", operands, attrs);
}

static Value createVectorAddReduction(OpBuilder &builder, Location loc,
                                      Value vectorValue) {
  return builder
      .create<vector::ReductionOp>(loc, vector::CombiningKind::ADD, vectorValue)
      .getResult();
}

static Value createVectorMaxReduction(OpBuilder &builder, Location loc,
                                      Value vectorValue) {
  return builder
      .create<vector::ReductionOp>(loc, vector::CombiningKind::MAXNUMF,
                                   vectorValue)
      .getResult();
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
  bool isScalarOffset = false;
};

struct CommonPlan {
  SmallVector<Operation *, 3> programIds;
  Operation *makeRange = nullptr;
  Value tailMaskValue;
  Value staticFullMaskValue;
  BlockArgument sizeArg;
  DenseMap<Operation *, int64_t> launchIdentityAxes;
  int64_t gridRank = 1;
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
      return funcOp->emitOpError(
          "expected tt.func to implement FunctionOpInterface");
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
  int64_t getGridRank() const { return common.gridRank; }
  StringRef getTailBoundArgName() const {
    auto it = argIndex.find(common.sizeArg);
    if (it == argIndex.end())
      return {};
    return args[it->second].valueName;
  }

  std::optional<int64_t> getLaunchIdentityAxis(Operation *op) const {
    auto it = common.launchIdentityAxes.find(op);
    if (it == common.launchIdentityAxes.end())
      return std::nullopt;
    return it->second;
  }

  bool isLaunchIdentityOp(Operation *op) const {
    return common.launchIdentityAxes.contains(op);
  }

  TTIRArgInfo *lookupArg(BlockArgument arg) {
    auto it = argIndex.find(arg);
    if (it == argIndex.end())
      return nullptr;
    return &args[it->second];
  }

  Operation *lookupDef(Value value) const { return value.getDefiningOp(); }

  bool isTailMaskValue(Value value) const {
    return value == common.tailMaskValue;
  }
  bool isRequiredDataValue(Value value) const {
    return requiredDataValues.contains(value);
  }
  const SmallVector<Value, 4> *
  lookupContiguousOffsetScalarTerms(Value value) const {
    auto it = contiguousOffsetScalarTerms.find(value);
    if (it == contiguousOffsetScalarTerms.end())
      return nullptr;
    return &it->second;
  }
  bool hasRequiredResult(Operation *op) const {
    if (!op)
      return false;
    for (Value result : op->getResults())
      if (requiredDataValues.contains(result))
        return true;
    return false;
  }

  bool isCanonicalInfrastructure(Operation *op) const {
    if ((isLaunchIdentityOp(op) && hasName(op, kTTGetProgramIdOpName)) ||
        op == common.makeRange)
      return true;
    if (op->getNumResults() == 0)
      return false;
    for (Value result : op->getResults()) {
      if (requiredDataValues.contains(result))
        return false;
      if (result == common.tailMaskValue ||
          result == common.staticFullMaskValue ||
          canonicalInfrastructureValues.contains(result))
        return true;
    }
    return false;
  }

  FailureOr<PointerExpr> classifyPointer(Value ptrValue) const {
    if (auto found = pointerExprs.find(ptrValue); found != pointerExprs.end())
      return found->second;

    if (auto blockArg = llvm::dyn_cast<BlockArgument>(ptrValue)) {
      auto it = argIndex.find(blockArg);
      if (it == argIndex.end() || !args[it->second].isPointer)
        return emitStagedDiagnostic(
            funcOp, "scalar pointer base not backed by a pointer argument");
      PointerExpr expr;
      expr.sourcePointerArg = blockArg;
      expr.element = args[it->second].pointerElement;
      return expr;
    }

    Operation *def = ptrValue.getDefiningOp();
    if (!def)
      return failure();

    if (hasName(def, kTTAddPtrOpName)) {
      if (def->getNumOperands() != 2)
        return def->emitOpError("expected tt.addptr with pointer and offset");
      FailureOr<PointerExpr> base = classifyPointer(def->getOperand(0));
      if (failed(base))
        return failure();
      bool scalarPointerResult = isScalarPointer(def->getResult(0).getType());
      bool tensorPointerResult =
          isRankedTensorPointer(def->getResult(0).getType());
      if (scalarPointerResult) {
        if (!contiguousOffsetScalarTerms.contains(def->getOperand(1)))
          return emitStagedDiagnostic(def,
                                      "non-canonical scalar pointer offset");
      } else if (tensorPointerResult) {
        if (!contiguousOffsetScalarTerms.contains(def->getOperand(1)))
          return emitStagedDiagnostic(def, "non-canonical pointer offset");
      } else {
        return emitStagedDiagnostic(def, "unsupported pointer expression");
      }
      PointerExpr expr = *base;
      expr.hasCanonicalOffset = true;
      expr.isScalarOffset = scalarPointerResult;
      expr.offsetValue = def->getOperand(1);
      if (base->hasCanonicalOffset) {
        const SmallVector<Value, 4> *baseTerms =
            lookupContiguousOffsetScalarTerms(base->offsetValue);
        const SmallVector<Value, 4> *offsetTerms =
            lookupContiguousOffsetScalarTerms(def->getOperand(1));
        if (!baseTerms || !offsetTerms)
          return emitStagedDiagnostic(def, "non-canonical pointer offset");
        SmallVector<Value, 4> combinedTerms;
        combinedTerms.append(baseTerms->begin(), baseTerms->end());
        combinedTerms.append(offsetTerms->begin(), offsetTerms->end());
        contiguousOffsetScalarTerms[def->getResult(0)] =
            std::move(combinedTerms);
        canonicalInfrastructureValues.insert(def->getResult(0));
        expr.offsetValue = def->getResult(0);
      }
      pointerExprs[ptrValue] = expr;
      return expr;
    }

    if (hasName(def, kTTSplatOpName) && def->getNumOperands() == 1) {
      Value scalar = def->getOperand(0);
      if (!isScalarPointer(scalar.getType()) ||
          !isRankedTensorPointer(def->getResult(0).getType()))
        return emitStagedDiagnostic(def, "unsupported pointer splat");
      FailureOr<PointerExpr> expr = classifyPointer(scalar);
      if (failed(expr))
        return failure();
      pointerExprs[ptrValue] = *expr;
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
          return emitStagedDiagnostic(
              funcOp, "block pointer or tensor pointer argument");
        if (!pointer->isSupportedPhase75Element)
          return emitUnsupportedPointerElementDiagnostic(funcOp,
                                                        pointer->pointeeType);
        info.isPointer = true;
        info.pointerElement = pointer->pointeeKind;
      } else if (isScalarI32(arg.getType()) || isScalarF32(arg.getType())) {
        info.role = isScalarF32(arg.getType()) ? TTIRArgumentRole::ScalarF32
                                               : TTIRArgumentRole::ScalarI32;
      } else {
        if (auto pointer =
                typeAdapter.classifyTensorPointerElement(arg.getType()))
          return emitUnsupportedPointerElementDiagnostic(funcOp,
                                                        pointer->pointeeType);
        return emitStagedDiagnostic(funcOp,
                                    "unsupported TTIR function argument type");
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
        if (hasName(user, kTTLoadOpName) && user->getNumOperands() == 3 &&
            user->getOperand(2) == result && isZeroLikeConstant(op))
          continue;
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
    Operation *stagedQKScoreGeneration = nullptr;
    TTIRReduceClassification stagedReduceClassification =
        TTIRReduceClassification::NotReduceCall;
    funcOp->walk([&](Operation *op) {
      if (stagedBodyFeature || stagedQKScoreGeneration)
        return WalkResult::interrupt();
      if (hasName(op, kTTCallOpName)) {
        if (isTTIRGeneratedScoreReductionCall(op)) {
          stagedQKScoreGeneration = op;
          return WalkResult::interrupt();
        }
        TTIRReduceCallInfo reduceInfo = classifyTTIRReduceCall(op);
        if (reduceInfo.classification ==
                TTIRReduceClassification::RankGreaterThanOne ||
            reduceInfo.classification == TTIRReduceClassification::NonAdd ||
            reduceInfo.classification ==
                TTIRReduceClassification::MultiResult ||
            reduceInfo.classification ==
                TTIRReduceClassification::Unsupported) {
          stagedBodyFeature = op;
          stagedReduceClassification = reduceInfo.classification;
          return WalkResult::interrupt();
        }
      }
      if (hasName(op, kTTDotOpName) || hasName(op, kTTReduceOpName) ||
          hasName(op, kTTReduceReturnOpName) ||
          hasName(op, kTTMakeBlockPtrOpName) || hasName(op, kTTAdvanceOpName)) {
        stagedBodyFeature = op;
        return WalkResult::interrupt();
      }
      if (isTTIRLoopCarriedF32ReduceAccumulation(op)) {
        stagedBodyFeature = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (stagedQKScoreGeneration)
      return stagedQKScoreGeneration->emitOpError()
             << "QK score generation is staged; READY_FOR_TRITON remains NO";
    if (stagedBodyFeature) {
      if (hasName(stagedBodyFeature, kTTCallOpName)) {
        if (stagedReduceClassification ==
            TTIRReduceClassification::RankGreaterThanOne)
          return stagedBodyFeature->emitOpError()
                 << "rank>1 tt.reduce is staged; READY_FOR_TRITON remains NO";
        if (stagedReduceClassification == TTIRReduceClassification::NonAdd)
          return stagedBodyFeature->emitOpError()
                 << "non-add tt.reduce is staged; READY_FOR_TRITON remains NO";
        if (stagedReduceClassification == TTIRReduceClassification::MultiResult)
          return stagedBodyFeature->emitOpError()
                 << "unsupported tt.reduce combiner; READY_FOR_TRITON remains "
                    "NO";
        return stagedBodyFeature->emitOpError()
               << "unsupported tt.reduce combiner; READY_FOR_TRITON remains NO";
      }
      if (hasName(stagedBodyFeature, kTTDotOpName))
        return emitStagedBodyFeatureDiagnostic(stagedBodyFeature,
                                               "tt.dot / contract");
      if (hasName(stagedBodyFeature, kTTReduceOpName) ||
          hasName(stagedBodyFeature, kTTReduceReturnOpName))
        return emitStagedBodyFeatureDiagnostic(stagedBodyFeature, "tt.reduce");
      if (isTTIRLoopCarriedF32ReduceAccumulation(stagedBodyFeature))
        return stagedBodyFeature->emitOpError()
               << "multi-block K accumulation is staged by body feature, not "
                  "unsupported control flow; READY_FOR_TRITON remains NO";
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
      if (hasName(op, kTTGetProgramIdOpName) ||
          hasName(op, kTTGetNumProgramsOpName)) {
        if (failed(attrAdapter.classifyProgramAxis(op))) {
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
          (void)emitStagedDiagnostic(op, "rank-2 or higher TTIR tensor result");
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

    if (programIds.empty() || programIds.size() > 3)
      return emitStagedDiagnostic(funcOp,
                                  "TTIR functions without 1..3 program_id ops");
    DenseSet<int64_t> seenProgramIdAxes;
    for (Operation *programId : programIds) {
      FailureOr<TTIRProgramAxis> axis =
          attrAdapter.classifyProgramAxis(programId);
      if (failed(axis))
        return failure();
      int64_t axisValue = static_cast<int64_t>(*axis);
      if (axisValue < 0 || axisValue > 2)
        return emitStagedDiagnostic(programId, "program_id axis outside 0..2");
      if (!seenProgramIdAxes.insert(axisValue).second)
        return emitStagedDiagnostic(programId, "duplicate program_id axis");
      common.programIds.push_back(programId);
      common.launchIdentityAxes[programId] = axisValue;
      common.gridRank = std::max(common.gridRank, axisValue + 1);
    }

    funcOp->walk([&](Operation *op) {
      if (!hasName(op, kTTGetNumProgramsOpName))
        return WalkResult::advance();
      FailureOr<TTIRProgramAxis> axis = attrAdapter.classifyProgramAxis(op);
      if (failed(axis))
        return WalkResult::interrupt();
      int64_t axisValue = static_cast<int64_t>(*axis);
      common.launchIdentityAxes[op] = axisValue;
      common.gridRank = std::max(common.gridRank, axisValue + 1);
      return WalkResult::advance();
    });

    if (ranges.size() != 1)
      return emitStagedDiagnostic(funcOp, "tt.make_range other than 0..16");
    FailureOr<TTIRRangeInfo> range =
        attrAdapter.classifyMakeRange(ranges.front());
    if (failed(range))
      return failure();
    if (range->start != 0 || range->end != 16 || range->width != 16)
      return emitStagedDiagnostic(ranges.front(),
                                  "tt.make_range other than 0..16");
    common.makeRange = ranges.front();

    if (stores.empty())
      return emitStagedDiagnostic(funcOp, "TTIR elementwise V1 without store");
    if (addptrs.empty())
      return emitStagedDiagnostic(funcOp, "TTIR elementwise V1 without addptr");

    for (Operation *addptr : addptrs) {
      if (addptr->getNumOperands() != 2)
        return emitStagedDiagnostic(
            addptr, "tt.addptr without pointer and offset operands");
      if (isScalarPointer(addptr->getResult(0).getType())) {
        if (failed(verifyScalarOffset(addptr->getOperand(1))))
          return failure();
      } else {
        if (failed(verifyContiguousOffset(addptr->getOperand(1))))
          return failure();
      }
    }

    Value mask;
    bool maskIsTail = false;
    SmallVector<Operation *, 8> memoryOps;
    funcOp->walk([&](Operation *op) {
      if (hasName(op, kTTLoadOpName) || hasName(op, kTTStoreOpName))
        memoryOps.push_back(op);
    });
    for (Operation *memoryOp : memoryOps) {
      bool isLoad = hasName(memoryOp, kTTLoadOpName);
      bool isStore = hasName(memoryOp, kTTStoreOpName);
      unsigned operands = memoryOp->getNumOperands();
      if (isLoad) {
        if (operands == 1)
          continue;
        if (operands != 3)
          return emitStagedDiagnostic(memoryOp,
                                      "tt.load without pointer, mask, other=0");
        if (!isZeroLikeConstant(memoryOp->getOperand(2).getDefiningOp()))
          return emitStagedDiagnostic(memoryOp, "tt.load nonzero other value");
        Value candidateMask = memoryOp->getOperand(1);
        bool candidateIsTail = matchesCanonicalTailMask(candidateMask);
        bool candidateIsFull = matchesStaticFullMask(candidateMask);
        if (!candidateIsTail && !candidateIsFull)
          return emitStagedDiagnostic(memoryOp,
                                      "sparse or unknown tt.load memory mask");
        FailureOr<PointerExpr> ptr = classifyPointer(memoryOp->getOperand(0));
        if (failed(ptr))
          return failure();
        if (candidateIsTail &&
            !tailMaskCompatibleWithPointerOffset(candidateMask,
                                                 ptr->offsetValue))
          return emitStagedDiagnostic(memoryOp,
                                      "sparse or unknown tt.load memory mask");
        if (!mask) {
          mask = candidateMask;
          maskIsTail = candidateIsTail;
        } else if (mask != candidateMask || maskIsTail != candidateIsTail) {
          return emitStagedDiagnostic(memoryOp,
                                      "multiple distinct transfer tail masks");
        }
      } else if (isStore) {
        if (operands == 2)
          continue;
        if (operands != 3)
          return emitStagedDiagnostic(memoryOp,
                                      "tt.store without pointer, value, mask");
        if (!llvm::isa<RankedTensorType>(memoryOp->getOperand(1).getType())) {
          if (!memoryOp->getOperand(2).getType().isInteger(1))
            return emitStagedDiagnostic(memoryOp,
                                        "scalar tt.store mask is not i1");
          FailureOr<PointerExpr> ptr = classifyPointer(memoryOp->getOperand(0));
          if (failed(ptr))
            return failure();
          if (!ptr->isScalarOffset)
            return emitStagedDiagnostic(memoryOp,
                                        "scalar tt.store pointer offset");
          continue;
        }
        Value candidateMask = memoryOp->getOperand(2);
        bool candidateIsTail = matchesCanonicalTailMask(candidateMask);
        bool candidateIsFull = matchesStaticFullMask(candidateMask);
        if (!candidateIsTail && !candidateIsFull)
          return emitStagedDiagnostic(memoryOp,
                                      "sparse or unknown tt.store memory mask");
        FailureOr<PointerExpr> ptr = classifyPointer(memoryOp->getOperand(0));
        if (failed(ptr))
          return failure();
        if (candidateIsTail &&
            !tailMaskCompatibleWithPointerOffset(candidateMask,
                                                 ptr->offsetValue))
          return emitStagedDiagnostic(memoryOp,
                                      "sparse or unknown tt.store memory mask");
        if (!mask) {
          mask = candidateMask;
          maskIsTail = candidateIsTail;
        } else if (mask != candidateMask || maskIsTail != candidateIsTail) {
          return emitStagedDiagnostic(memoryOp,
                                      "multiple distinct transfer tail masks");
        }
      }
    }
    if (mask) {
      if (maskIsTail) {
        if (failed(recordCanonicalTailMask(mask)))
          return failure();
        common.tailMaskValue = mask;
      } else {
        common.staticFullMaskValue = mask;
        canonicalInfrastructureValues.insert(mask);
      }
    }
    return success();
  }

  LogicalResult verifyContiguousOffset(Value offset) {
    if (contiguousOffsetScalarTerms.contains(offset))
      return success();
    SmallVector<Value, 4> scalarTerms;
    bool sawRange = false;
    DenseSet<Value> visiting;
    if (failed(collectContiguousOffsetTerms(offset, scalarTerms, sawRange,
                                            visiting)))
      return failure();
    if (!sawRange)
      return emitStagedDiagnostic(
          offset.getDefiningOp() ? offset.getDefiningOp() : funcOp,
          "pointer offset missing tt.make_range lanes");
    for (Value term : scalarTerms)
      markRequiredValue(term);
    contiguousOffsetScalarTerms[offset] = std::move(scalarTerms);
    canonicalInfrastructureValues.insert(offset);
    return success();
  }

  LogicalResult verifyScalarOffset(Value offset) {
    if (contiguousOffsetScalarTerms.contains(offset))
      return success();
    if (!offset || (!offset.getType().isSignlessInteger(32) &&
                    !offset.getType().isIndex()))
      return emitStagedDiagnostic(
          offset.getDefiningOp() ? offset.getDefiningOp() : funcOp,
          "non-canonical scalar pointer offset");
    if (valueDependsOnMakeRange(offset))
      return emitStagedDiagnostic(
          offset.getDefiningOp() ? offset.getDefiningOp() : funcOp,
          "scalar pointer offset depends on lanes");
    SmallVector<Value, 1> scalarTerms;
    scalarTerms.push_back(offset);
    markRequiredValue(offset);
    contiguousOffsetScalarTerms[offset] = std::move(scalarTerms);
    canonicalInfrastructureValues.insert(offset);
    return success();
  }

  LogicalResult
  collectContiguousOffsetTerms(Value value, SmallVectorImpl<Value> &scalarTerms,
                               bool &sawRange, DenseSet<Value> &visiting) {
    if (!value)
      return emitStagedDiagnostic(funcOp, "non-canonical pointer offset");
    if (value == common.makeRange->getResult(0)) {
      sawRange = true;
      canonicalInfrastructureValues.insert(value);
      return success();
    }
    if (auto cached = lookupContiguousOffsetScalarTerms(value)) {
      scalarTerms.append(cached->begin(), cached->end());
      sawRange = true;
      return success();
    }
    if (!visiting.insert(value).second)
      return emitStagedDiagnostic(value.getDefiningOp() ? value.getDefiningOp()
                                                        : funcOp,
                                  "cyclic pointer offset expression");
    Operation *def = value.getDefiningOp();
    if (!def) {
      visiting.erase(value);
      return emitStagedDiagnostic(funcOp,
                                  "pointer offset missing tt.make_range lanes");
    }

    if (hasName(def, kTTSplatOpName) && def->getNumOperands() == 1) {
      Value scalar = def->getOperand(0);
      if (!isScalarI32(scalar.getType()) && !scalar.getType().isIndex()) {
        visiting.erase(value);
        return emitStagedDiagnostic(def,
                                    "pointer scalar base is not i32/index");
      }
      scalarTerms.push_back(scalar);
      canonicalInfrastructureValues.insert(value);
      visiting.erase(value);
      return success();
    }

    if (hasName(def, "arith.addi") && def->getNumOperands() == 2) {
      if (failed(collectContiguousOffsetTerms(def->getOperand(0), scalarTerms,
                                              sawRange, visiting)) ||
          failed(collectContiguousOffsetTerms(def->getOperand(1), scalarTerms,
                                              sawRange, visiting))) {
        visiting.erase(value);
        return failure();
      }
      canonicalInfrastructureValues.insert(value);
      visiting.erase(value);
      return success();
    }

    if (hasName(def, "arith.muli") && def->getNumOperands() == 2) {
      bool lhsLane = valueDependsOnMakeRange(def->getOperand(0));
      bool rhsLane = valueDependsOnMakeRange(def->getOperand(1));
      if (lhsLane || rhsLane) {
        Value laneOperand = lhsLane ? def->getOperand(0) : def->getOperand(1);
        visiting.erase(value);
        if (laneOperand == common.makeRange->getResult(0))
          return emitStagedDiagnostic(
              def, "lane-varying stride/gather pointer expression staged");
        return emitStagedDiagnostic(
            def, "non-contiguous column slice pointer expression staged");
      }
    }

    visiting.erase(value);
    return emitStagedDiagnostic(def, "non-canonical pointer offset");
  }

  bool valueDependsOnMakeRange(Value value) const {
    DenseSet<Value> visiting;
    return valueDependsOnMakeRange(value, visiting);
  }

  bool valueDependsOnMakeRange(Value value, DenseSet<Value> &visiting) const {
    if (!value)
      return false;
    if (value == common.makeRange->getResult(0))
      return true;
    if (!visiting.insert(value).second)
      return false;
    Operation *def = value.getDefiningOp();
    if (!def) {
      visiting.erase(value);
      return false;
    }
    for (Value operand : def->getOperands())
      if (valueDependsOnMakeRange(operand, visiting)) {
        visiting.erase(value);
        return true;
      }
    visiting.erase(value);
    return false;
  }

  std::optional<BlockArgument> matchCanonicalTailMask(Value mask) {
    Operation *cmp = mask.getDefiningOp();
    auto cmpi = llvm::dyn_cast_or_null<arith::CmpIOp>(cmp);
    if (!cmpi)
      return std::nullopt;
    if (cmpi.getPredicate() != arith::CmpIPredicate::slt)
      return std::nullopt;
    if (failed(verifyContiguousOffset(cmpi.getOperand(0))))
      return std::nullopt;
    const SmallVector<Value, 4> *maskTerms =
        lookupContiguousOffsetScalarTerms(cmpi.getOperand(0));
    if (!maskTerms)
      return std::nullopt;

    Operation *rhsSplat = cmpi.getOperand(1).getDefiningOp();
    if (!hasName(rhsSplat, kTTSplatOpName) || rhsSplat->getNumOperands() != 1)
      return std::nullopt;
    auto sizeArg = llvm::dyn_cast<BlockArgument>(rhsSplat->getOperand(0));
    if (!sizeArg)
      return std::nullopt;
    auto it = argIndex.find(sizeArg);
    if (it == argIndex.end())
      return std::nullopt;
    if (!isScalarI32(args[it->second].sourceType))
      return std::nullopt;
    return sizeArg;
  }

  bool matchesCanonicalTailMask(Value mask) {
    return matchCanonicalTailMask(mask).has_value();
  }

  bool matchesStaticFullMask(Value mask) {
    Operation *cmp = mask.getDefiningOp();
    auto cmpi = llvm::dyn_cast_or_null<arith::CmpIOp>(cmp);
    if (!cmpi || cmpi.getPredicate() != arith::CmpIPredicate::slt)
      return false;
    if (failed(verifyContiguousOffset(cmpi.getOperand(0))))
      return false;
    auto constant =
        llvm::dyn_cast_or_null<arith::ConstantOp>(cmpi.getOperand(1).getDefiningOp());
    if (!constant)
      return false;
    auto dense = llvm::dyn_cast<DenseElementsAttr>(constant.getValue());
    if (!dense || !dense.getElementType().isSignlessInteger(32))
      return false;
    for (APInt value : dense.getValues<APInt>())
      if (value.getSExtValue() != static_cast<int64_t>(kPhase7VectorWidth))
        return false;
    return true;
  }

  bool tailMaskCompatibleWithPointerOffset(Value mask, Value pointerOffset) {
    Operation *cmp = mask.getDefiningOp();
    auto cmpi = llvm::dyn_cast_or_null<arith::CmpIOp>(cmp);
    if (!cmpi)
      return false;
    Value maskOffset = cmpi.getOperand(0);
    if (failed(verifyContiguousOffset(maskOffset)) ||
        failed(verifyContiguousOffset(pointerOffset)))
      return false;
    const SmallVector<Value, 4> *maskTerms =
        lookupContiguousOffsetScalarTerms(maskOffset);
    const SmallVector<Value, 4> *pointerTerms =
        lookupContiguousOffsetScalarTerms(pointerOffset);
    if (!maskTerms || !pointerTerms)
      return false;
    if (maskTerms->empty()) {
      for (Value term : *pointerTerms)
        if (!isRowStrideBaseTerm(term))
          return false;
      return true;
    }
    DenseSet<Value> pointerTermSet;
    for (Value term : *pointerTerms)
      pointerTermSet.insert(term);
    for (Value term : *maskTerms)
      if (!pointerTermSet.contains(term))
        return false;
    return true;
  }

  bool isRowStrideBaseTerm(Value term) const {
    Operation *def = term.getDefiningOp();
    if (!hasName(def, "arith.muli") || def->getNumOperands() != 2)
      return false;
    auto isLaunch = [&](Value value) {
      return plannerLaunchIdentityValue(value);
    };
    auto isDynamicStride = [&](Value value) {
      return llvm::isa<BlockArgument>(value);
    };
    return (isLaunch(def->getOperand(0)) &&
            isDynamicStride(def->getOperand(1))) ||
           (isLaunch(def->getOperand(1)) &&
            isDynamicStride(def->getOperand(0)));
  }

  bool plannerLaunchIdentityValue(Value value) const {
    Operation *def = value.getDefiningOp();
    return def && common.launchIdentityAxes.contains(def);
  }

  LogicalResult recordCanonicalTailMask(Value mask) {
    std::optional<BlockArgument> matchedSizeArg = matchCanonicalTailMask(mask);
    if (!matchedSizeArg)
      return emitStagedDiagnostic(
          mask.getDefiningOp() ? mask.getDefiningOp() : funcOp,
          "tail mask not formed by canonical offsets < bound");
    Operation *cmp = mask.getDefiningOp();
    Operation *rhsSplat =
        llvm::cast<arith::CmpIOp>(cmp).getOperand(1).getDefiningOp();
    BlockArgument sizeArg = *matchedSizeArg;
    auto it = argIndex.find(sizeArg);
    if (it == argIndex.end())
      return emitInternalError(
          rhsSplat, "tail mask size argument missing from arg table");
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
        (void)emitInternalError(op,
                                "pointer base argument missing from arg table");
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
        hasName(def, kTTGetProgramIdOpName) ||
        hasName(def, kTTGetNumProgramsOpName))
      return;
    if (def->getNumResults() == 1 &&
        (canonicalInfrastructureValues.contains(def->getResult(0)) ||
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
      if (hasName(op, kTTStoreOpName) && op->getNumOperands() >= 2) {
        markRequiredValue(op->getOperand(1));
        if (op->getNumOperands() == 3 &&
            !llvm::isa<RankedTensorType>(op->getOperand(1).getType()))
          markRequiredValue(op->getOperand(2));
      }
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
  mutable DenseMap<Value, SmallVector<Value, 4>> contiguousOffsetScalarTerms;
  mutable DenseSet<Value> canonicalInfrastructureValues;
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

    if (failed(
            lowerBlock(planner.getEntryBlock(), valueFunc.getBody().front())))
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
      return sourceFunc->emitOpError(
          "expected tt.func to implement FunctionOpInterface");

    SmallVector<Type, 8> argTypes;
    for (const TTIRArgInfo &arg : planner.getArgs()) {
      if (arg.isPointer) {
        Type elemType =
            typeAdapter.getValueElementType(arg.pointerElement, builder);
        if (!elemType)
          return sourceFunc->emitOpError("unsupported pointer element type");
        auto memorySpace = mlir::vc4value::GlobalMemorySpaceAttr::get(ctx);
        int64_t extent = planner.getCommonPlan().sizeArg
                             ? ShapedType::kDynamic
                             : static_cast<int64_t>(kPhase7VectorWidth);
        argTypes.push_back(MemRefType::get(
            {extent}, elemType, MemRefLayoutAttrInterface(), memorySpace));
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
    valueFunc->setAttr(kVC4ValueGridRankAttr,
                       builder.getI32IntegerAttr(planner.getGridRank()));
    if (functionNeedsApproxSFUMathPolicy())
      valueFunc->setAttr(kVC4ValueMathPolicyAttr,
                         builder.getStringAttr("approx_sfu"));
    if (functionNeedsPositiveFPDomain() && !functionNeedsFiniteFPDomain() &&
        !functionNeedsFiniteReductionPolicy())
      valueFunc->setAttr(kVC4ValueFPDomainAttr,
                         builder.getStringAttr("finite_positive"));
    else if (functionNeedsFiniteFPDomain() ||
             functionNeedsFiniteReductionPolicy())
      valueFunc->setAttr(kVC4ValueFPDomainAttr,
                         builder.getStringAttr("finite"));
    if (functionNeedsFiniteReductionPolicy())
      valueFunc->setAttr(kVC4ValueReductionPolicyAttr,
                         builder.getStringAttr("finite_tree"));
    if (functionNeedsFiniteMaxPolicy())
      valueFunc->setAttr(kVC4ValueMaxPolicyAttr,
                         builder.getStringAttr("finite"));
    if (functionNeedsFiniteF16StoragePolicy())
      valueFunc->setAttr(kVC4ValueF16StoragePolicyAttr,
                         builder.getStringAttr("finite"));

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
                               builder.getArrayAttr({builder.getStringAttr(
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
    planner.getEntryBlock().walk([&](Operation *op) {
      if (needs)
        return WalkResult::interrupt();
      if (llvm::isa<math::ExpOp>(op) || hasName(op, "arith.divf")) {
        needs = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return needs;
  }

  bool functionNeedsPositiveFPDomain() const {
    bool needs = false;
    planner.getEntryBlock().walk([&](Operation *op) {
      if (llvm::isa<math::LogOp, math::RsqrtOp, math::SqrtOp>(op)) {
        needs = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return needs;
  }

  bool functionNeedsApproxSFUMathPolicy() const {
    bool needs = false;
    planner.getEntryBlock().walk([&](Operation *op) {
      if (llvm::isa<math::ExpOp, math::LogOp, math::RsqrtOp, math::SqrtOp>(op) ||
          hasName(op, "arith.divf")) {
        needs = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return needs;
  }

  bool functionNeedsFiniteReductionPolicy() const {
    bool needs = false;
    planner.getEntryBlock().walk([&](Operation *op) {
      if (needs || !hasName(op, kTTCallOpName))
        return WalkResult::advance();
      TTIRReduceCallInfo reduceInfo = classifyTTIRReduceCall(op);
      if (reduceInfo.classification == TTIRReduceClassification::AddF32 ||
          reduceInfo.classification == TTIRReduceClassification::MaxF32) {
        needs = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return needs;
  }

  bool functionNeedsFiniteMaxPolicy() const {
    bool needs = false;
    planner.getEntryBlock().walk([&](Operation *op) {
      if (needs)
        return WalkResult::interrupt();
      if ((hasName(op, "arith.maxnumf") || hasName(op, "arith.maximumf")) &&
          op->getNumResults() == 1 && op->getResult(0).getType().isF32()) {
        needs = true;
        return WalkResult::interrupt();
      }
      if (hasName(op, kTTCallOpName)) {
        TTIRReduceCallInfo reduceInfo = classifyTTIRReduceCall(op);
        if (reduceInfo.classification == TTIRReduceClassification::MaxF32) {
          needs = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return needs;
  }

  bool functionNeedsFiniteF16StoragePolicy() const {
    for (const TTIRArgInfo &arg : planner.getArgs())
      if (arg.isPointer && arg.stored &&
          arg.pointerElement == TTIRScalarElementKind::F16)
        return true;
    return false;
  }

  LogicalResult emitCommonPrefix() {
    const CommonPlan &common = planner.getCommonPlan();
    for (Operation *programId : common.programIds) {
      std::optional<int64_t> axis = planner.getLaunchIdentityAxis(programId);
      if (!axis)
        return emitInternalError(programId,
                                 "program_id axis missing from plan");
      Location loc = programId->getLoc();
      Value pidIndex = createVC4ValueProgramId(builder, loc, *axis);
      Value pidI32 = builder.create<arith::IndexCastOp>(
          loc, builder.getI32Type(), pidIndex);
      bindValue(programId->getResult(0), pidI32);
    }
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
      pointerScopes.back().push_back(
          {source, found == pointerValues.end()
                       ? std::optional<PointerExpr>()
                       : std::optional<PointerExpr>(found->second)});
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

    for (auto it = pointerScopes.back().rbegin(),
              e = pointerScopes.back().rend();
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
        return emitInternalError(op, Twine("result-producing ") +
                                         op->getName().getStringRef() +
                                         " reported zero-result lowering");
      return success();
    case LoweringOutcome::IgnoredDeadProofOp:
      if (!planner.canIgnoreAsDeadProofOp(op))
        return emitInternalError(op, Twine("ignored-dead proof failed for ") +
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

    if (planner.isLaunchIdentityOp(op) && hasName(op, kTTGetProgramIdOpName))
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
    if (hasName(op, kTTCallOpName))
      return lowerTTCall(op);
    if (hasName(op, kTTGetNumProgramsOpName))
      return lowerTTGetNumPrograms(op);

    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "math")
      return lowerMath(op);
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
    bindValue(op->getResult(0), builder.create<arith::ConstantOp>(
                                    op->getLoc(), vectorType, attr));
    return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
  }

  LogicalResult lowerTTGetNumPrograms(Operation *op) {
    if (op->getNumResults() != 1)
      return emitStagedDiagnostic(op, "tt.get_num_programs result shape");
    FailureOr<TTIRProgramAxis> axis = TTIRAttrAdapter().classifyProgramAxis(op);
    if (failed(axis))
      return failure();
    int64_t axisValue = static_cast<int64_t>(*axis);
    if (axisValue < 0 || axisValue > 2)
      return emitStagedDiagnostic(op, "num_programs axis outside 0..2");
    Value numProgramsIndex =
        createVC4ValueNumPrograms(builder, op->getLoc(), axisValue);
    bindValue(op->getResult(0),
              builder.create<arith::IndexCastOp>(
                  op->getLoc(), builder.getI32Type(), numProgramsIndex));
    return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
  }

  LogicalResult lowerTTSplat(Operation *op) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return emitStagedDiagnostic(op, "non-unary tt.splat");

    if (isScalarPointer(op->getOperand(0).getType()) &&
        isRankedTensorPointer(op->getResult(0).getType())) {
      FailureOr<PointerExpr> ptr = planner.classifyPointer(op->getResult(0));
      if (failed(ptr))
        return failure();
      PointerExpr lowered = *ptr;
      auto found = pointerValues.find(ptr->sourcePointerArg);
      if (found == pointerValues.end())
        return emitInternalError(op,
                                 "pointer splat source memref not available");
      lowered.valueMemref = found->second.valueMemref;
      bindPointer(op->getResult(0), lowered);
      return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
    }

    Value src = op->getOperand(0);
    Type resultType = typeAdapter.convertTensorToValueVector(
        op->getResult(0).getType(), builder);
    if (!resultType)
      return emitStagedDiagnostic(op, "tt.splat result type");
    Value scalar = lookup(src);
    if (!scalar)
      return emitStagedDiagnostic(op,
                                  "tt.splat source not available in value IR");
    if (scalar.getType().isIndex()) {
      auto vectorType = llvm::dyn_cast<VectorType>(resultType);
      if (!vectorType || !vectorType.getElementType().isSignlessInteger(32))
        return emitStagedDiagnostic(op, "tt.splat source/result type");
      scalar = builder.create<arith::IndexCastOp>(
          op->getLoc(), builder.getI32Type(), scalar);
    }
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
    if (op->getNumResults() == 1 &&
        !planner.isRequiredDataValue(op->getResult(0)))
      return emitStagedDiagnostic(
          op, "dead tt.load outside ignored-dead proof whitelist");
    if ((op->getNumOperands() != 1 && op->getNumOperands() != 3) ||
        op->getNumResults() != 1)
      return emitStagedDiagnostic(op, "tt.load without pointer, mask, other=0");
    if (op->getNumOperands() == 3) {
      Operation *other = op->getOperand(2).getDefiningOp();
      if (!isZeroLikeConstant(other))
        return emitStagedDiagnostic(op, "tt.load nonzero other value");
    }

    FailureOr<PointerExpr> ptr = planner.classifyPointer(op->getOperand(0));
    if (failed(ptr))
      return failure();
    if (!ptr->hasCanonicalOffset)
      return emitStagedDiagnostic(
          op, "tt.load pointer without canonical addptr offset");

    Type resultType = typeAdapter.convertTensorToValueVector(
        op->getResult(0).getType(), builder);
    auto vectorType = llvm::dyn_cast_or_null<VectorType>(resultType);
    if (!vectorType)
      return emitStagedDiagnostic(op, "scalar tt.load");
    Value memref = pointerValues[ptr->sourcePointerArg].valueMemref;
    Value pad = createZeroPadding(op->getLoc(), vectorType.getElementType());
    FailureOr<Value> transferIndex = ensureTransferIndex(ptr->offsetValue, op);
    if (failed(transferIndex))
      return failure();
    Value mask;
    if (op->getNumOperands() == 3) {
      FailureOr<Value> maybeMask = ensureMemoryMask(
          op->getOperand(1), op, "sparse or unknown tt.load memory mask");
      if (failed(maybeMask))
        return failure();
      mask = *maybeMask;
    }
    bindValue(op->getResult(0),
              createVectorTransferRead(builder, op->getLoc(), memref,
                                       *transferIndex, pad, mask, vectorType));
    return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
  }

  LogicalResult lowerTTStore(Operation *op) {
    if (op->getNumOperands() != 2 && op->getNumOperands() != 3)
      return emitStagedDiagnostic(op, "tt.store without pointer, value, mask");
    FailureOr<PointerExpr> ptr = planner.classifyPointer(op->getOperand(0));
    if (failed(ptr))
      return failure();
    if (!ptr->hasCanonicalOffset)
      return emitStagedDiagnostic(
          op, "tt.store pointer without canonical addptr offset");
    Value value = lookup(op->getOperand(1));
    if (!value)
      return emitStagedDiagnostic(
          op, "tt.store value was not produced by supported compute DAG");
    Value memref = pointerValues[ptr->sourcePointerArg].valueMemref;
    FailureOr<Value> transferIndex = ensureTransferIndex(ptr->offsetValue, op);
    if (failed(transferIndex))
      return failure();
    if (!llvm::isa<RankedTensorType>(op->getOperand(1).getType())) {
      if (!isScalarI32(value.getType()) && !isScalarF32(value.getType()))
        return emitStagedDiagnostic(op,
                                    "scalar tt.store unsupported value type");
      if (!ptr->isScalarOffset)
        return emitStagedDiagnostic(op, "scalar tt.store pointer offset");
      auto createStore = [&]() {
        builder.create<memref::StoreOp>(op->getLoc(), value, memref,
                                        ValueRange{*transferIndex});
      };
      if (op->getNumOperands() == 3) {
        Value mask = lookup(op->getOperand(2));
        if (!mask || !mask.getType().isInteger(1))
          return emitStagedDiagnostic(op, "scalar tt.store mask not available");
        auto ifOp = builder.create<scf::IfOp>(op->getLoc(), mask,
                                              /*withElseRegion=*/false);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        createStore();
        builder.setInsertionPointAfter(ifOp);
      } else {
        createStore();
      }
      return finishLowering(op, LoweringOutcome::LoweredZeroResult);
    }
    Value mask;
    if (op->getNumOperands() == 3) {
      FailureOr<Value> maybeMask = ensureMemoryMask(
          op->getOperand(2), op, "sparse or unknown tt.store memory mask");
      if (failed(maybeMask))
        return failure();
      mask = *maybeMask;
    }
    createVectorTransferWrite(builder, op->getLoc(), value, memref,
                              *transferIndex, mask);
    return finishLowering(op, LoweringOutcome::LoweredZeroResult);
  }

  LogicalResult lowerTTCall(Operation *op) {
    TTIRReduceCallInfo reduceInfo = classifyTTIRReduceCall(op);
    switch (reduceInfo.classification) {
    case TTIRReduceClassification::AddI32:
    case TTIRReduceClassification::AddF32:
    case TTIRReduceClassification::MaxF32:
      break;
    case TTIRReduceClassification::RankGreaterThanOne:
      return op->emitOpError()
             << "rank>1 tt.reduce is staged; READY_FOR_TRITON remains NO";
    case TTIRReduceClassification::NonAdd:
      return op->emitOpError()
             << "non-add tt.reduce is staged; READY_FOR_TRITON remains NO";
    case TTIRReduceClassification::MultiResult:
    case TTIRReduceClassification::Unsupported:
      return op->emitOpError()
             << "unsupported tt.reduce combiner; READY_FOR_TRITON remains NO";
    case TTIRReduceClassification::NotReduceCall:
      return emitStagedDiagnostic(op, "unsupported tt.call");
    }

    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return op->emitOpError()
             << "unsupported tt.reduce combiner; READY_FOR_TRITON remains NO";
    Value input = lookup(op->getOperand(0));
    if (!input)
      return emitStagedDiagnostic(op, "tt.reduce input not available");
    auto vectorType = llvm::dyn_cast<VectorType>(input.getType());
    if (!vectorType || vectorType.getRank() != 1 ||
        vectorType.getDimSize(0) != kPhase7VectorWidth)
      return op->emitOpError()
             << "rank>1 tt.reduce is staged; READY_FOR_TRITON remains NO";
    Type elementType = vectorType.getElementType();
    if (!elementType.isF32() && !elementType.isSignlessInteger(32))
      return op->emitOpError()
             << "unsupported tt.reduce combiner; READY_FOR_TRITON remains NO";
    if (reduceInfo.classification == TTIRReduceClassification::AddF32 ||
        reduceInfo.classification == TTIRReduceClassification::MaxF32) {
      valueFunc->setAttr(kVC4ValueFPDomainAttr,
                         builder.getStringAttr("finite"));
      valueFunc->setAttr(kVC4ValueReductionPolicyAttr,
                         builder.getStringAttr("finite_tree"));
      if (reduceInfo.classification == TTIRReduceClassification::MaxF32)
        valueFunc->setAttr(kVC4ValueMaxPolicyAttr,
                           builder.getStringAttr("finite"));
    }
    bindValue(op->getResult(0),
              reduceInfo.classification == TTIRReduceClassification::MaxF32
                  ? createVectorMaxReduction(builder, op->getLoc(), input)
                  : createVectorAddReduction(builder, op->getLoc(), input));
    return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
  }

  LogicalResult lowerArith(Operation *op) {
    if (auto constant = llvm::dyn_cast<arith::ConstantOp>(op))
      return lowerArithConstant(constant);
    if (hasName(op, "arith.sitofp"))
      return op->emitOpError()
             << "i32 to f32 numeric cast staged by lower-half gap; "
             << "READY_FOR_TRITON remains NO";
    if (hasName(op, "arith.fptosi") || hasName(op, "arith.fptoui"))
      return op->emitOpError()
             << "fp-to-int numeric cast is staged; "
             << "READY_FOR_TRITON remains NO";
    if (hasName(op, "arith.bitcast")) {
      if (op->getNumOperands() != 1 || op->getNumResults() != 1)
        return emitStagedDiagnostic(op, "non-unary arith.bitcast");
      Value mapped = lookup(op->getOperand(0));
      if (!mapped)
        return emitStagedDiagnostic(op, "arith.bitcast operand not available");
      Type resultType = convertResultType(op->getResult(0).getType());
      if (!resultType)
        return emitStagedDiagnostic(op, "arith.bitcast changing value type");
      if (resultType != mapped.getType()) {
        if (!(op->getOperand(0).getType().isSignlessInteger(32) &&
              op->getResult(0).getType().isSignlessInteger(32) &&
              mapped.getType().isIndex()))
          return emitStagedDiagnostic(op, "arith.bitcast changing value type");
      }
      if (resultType != mapped.getType() && !mapped.getType().isIndex())
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
      FailureOr<Value> mask = ensureTailMask(
          op->getResult(0), op, "sparse or unknown TTIR memory mask");
      if (failed(mask))
        return failure();
      bindValue(op->getResult(0), *mask);
      return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
    }

    static constexpr StringRef supportedArithOps[] = {
        "arith.addf",  "arith.subf", "arith.mulf",  "arith.divf",  "arith.extf",
        "arith.truncf", "arith.addi", "arith.subi",  "arith.muli",
        "arith.cmpf",  "arith.cmpi", "arith.select", "arith.maxnumf",
        "arith.maximumf"};
    if (!hasAnyName(op, supportedArithOps))
      return emitStagedDiagnostic(op, op->getName().getStringRef());

    if (hasAnyName(op, {"arith.addf", "arith.subf", "arith.mulf",
                        "arith.cmpf"})) {
      for (Type type : op->getOperandTypes())
        if (isScalarF16(getStorageScalarElement(type)))
          return op->emitOpError()
                 << "native f16 arithmetic is staged; "
                 << "READY_FOR_TRITON remains NO";
      for (Type type : op->getResultTypes())
        if (isScalarF16(getStorageScalarElement(type)))
          return op->emitOpError()
                 << "native f16 arithmetic is staged; "
                 << "READY_FOR_TRITON remains NO";
    }

    SmallVector<Value, 4> operands;
    operands.reserve(op->getNumOperands());
    for (Value operand : op->getOperands()) {
      Value mapped = lookup(operand);
      if (!mapped)
        return emitStagedDiagnostic(
            op, "arith operand not available in supported value dataflow");
      operands.push_back(mapped);
    }

    SmallVector<Type, 2> resultTypes;
    for (Type type : op->getResultTypes()) {
      Type mapped = convertResultType(type);
      if (!mapped)
        return emitStagedDiagnostic(op, "arith result type");
      resultTypes.push_back(mapped);
    }
    if (op->getNumResults() == 1 &&
        hasAnyName(op, {"arith.addi", "arith.subi", "arith.muli"}) &&
        (resultTypes[0].isIndex() || resultTypes[0].isSignlessInteger(32))) {
      for (Value &operand : operands) {
        Type operandType = operand.getType();
        if (operandType == resultTypes[0])
          continue;
        if (!((operandType.isIndex() || operandType.isSignlessInteger(32)) &&
              (resultTypes[0].isIndex() ||
               resultTypes[0].isSignlessInteger(32))))
          continue;
        operand = builder.create<arith::IndexCastOp>(op->getLoc(),
                                                     resultTypes[0], operand);
      }
    }

    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    for (NamedAttribute attr : op->getAttrs())
      state.addAttribute(attr.getName(), attr.getValue());
    if (hasName(op, "arith.divf")) {
      state.addAttribute(kVC4ValueMathPolicyAttr,
                         builder.getStringAttr("approx_sfu"));
      state.addAttribute(kVC4ValueFPDomainAttr,
                         builder.getStringAttr("finite"));
    }
    if (hasName(op, "arith.maxnumf") || hasName(op, "arith.maximumf")) {
      state.addAttribute(kVC4ValueFPDomainAttr, builder.getStringAttr("finite"));
      state.addAttribute(kVC4ValueMaxPolicyAttr, builder.getStringAttr("finite"));
      valueFunc->setAttr(kVC4ValueFPDomainAttr, builder.getStringAttr("finite"));
      valueFunc->setAttr(kVC4ValueMaxPolicyAttr, builder.getStringAttr("finite"));
    }
    Operation *created = builder.create(state);
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), created->getResults()))
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
    bindValue(op.getResult(), builder.create<arith::ConstantOp>(
                                  op.getLoc(), resultType, typedAttr));
    return finishLowering(op.getOperation(),
                          LoweringOutcome::LoweredWithResultsBound);
  }

  LogicalResult lowerMath(Operation *op) {
    if (!llvm::isa<math::ExpOp, math::LogOp, math::RsqrtOp, math::SqrtOp>(op))
      return emitStagedDiagnostic(op, op->getName().getStringRef());
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return emitStagedDiagnostic(op, "math operation arity");

    Value operand = lookup(op->getOperand(0));
    if (!operand)
      return emitStagedDiagnostic(op,
                                  "math operand not available in value IR");
    Type resultType = convertResultType(op->getResult(0).getType());
    if (!resultType)
      return emitStagedDiagnostic(op, "math result type");

    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands(operand);
    state.addTypes(resultType);
    state.addAttribute(kVC4ValueMathPolicyAttr,
                       builder.getStringAttr("approx_sfu"));
    state.addAttribute(kVC4ValueFPDomainAttr,
                       builder.getStringAttr(
                           llvm::isa<math::LogOp, math::RsqrtOp, math::SqrtOp>(
                               op)
                               ? "finite_positive"
                               : "finite"));
    Operation *created = builder.create(state);
    bindValue(op->getResult(0), created->getResult(0));
    return finishLowering(op, LoweringOutcome::LoweredWithResultsBound);
  }

  Type convertResultType(Type type) {
    if (type.isIndex() || type.isInteger(1) || type.isSignlessInteger(32) ||
        type.isF16() || type.isF32())
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
        auto override = regionArgTypeOverrides.find(arg);
        Type converted =
            override == regionArgTypeOverrides.end()
                ? convertRegionValueType(arg.getType())
                : override->second;
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
        return emitPermanentReject(op,
                                   "vector/per-lane branch condition as CFG");
      return lowerGenericTerminator(op);
    }
    if (!hasAnyName(op, {"scf.if", "scf.for", "scf.while"}))
      return emitStagedDiagnostic(op, op->getName().getStringRef());

    if (hasName(op, "scf.if") && (op->getNumOperands() != 1 ||
                                  !op->getOperand(0).getType().isInteger(1)))
      return emitPermanentReject(op, "vector/per-lane branch condition as CFG");

    SmallVector<Value, 8> operands;
    operands.reserve(op->getNumOperands());
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      FailureOr<Value> mapped =
          hasName(op, "scf.for") && index < 3
              ? ensureIndexValue(operand, op, "SCF operand not available in value map")
              : lookupValue(operand, op, "SCF operand not available in value map");
      if (failed(mapped))
        return failure();
      operands.push_back(*mapped);
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

    SmallVector<std::pair<BlockArgument, Type>, 1> savedOverrides;
    if (hasName(op, "scf.for") && op->getNumRegions() == 1 &&
        !op->getRegion(0).empty() &&
        op->getRegion(0).front().getNumArguments() >= 1) {
      BlockArgument sourceIv = op->getRegion(0).front().getArgument(0);
      savedOverrides.push_back({sourceIv, regionArgTypeOverrides[sourceIv]});
      regionArgTypeOverrides[sourceIv] = builder.getIndexType();
    }
    for (auto [sourceRegion, destRegion] :
         llvm::zip(op->getRegions(), created->getRegions())) {
      if (failed(lowerRegion(sourceRegion, destRegion))) {
        for (auto [arg, oldType] : savedOverrides) {
          if (oldType)
            regionArgTypeOverrides[arg] = oldType;
          else
            regionArgTypeOverrides.erase(arg);
        }
        return failure();
      }
    }
    for (auto [arg, oldType] : savedOverrides) {
      if (oldType)
        regionArgTypeOverrides[arg] = oldType;
      else
        regionArgTypeOverrides.erase(arg);
    }
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
      return emitStagedDiagnostic(
          op, "cf.switch was not emitted by Phase85a2 TTIR");
    if (!hasAnyName(op, {"cf.br", "cf.cond_br"}))
      return emitStagedDiagnostic(op, op->getName().getStringRef());
    if (hasName(op, "cf.cond_br") &&
        (op->getNumOperands() < 1 || !op->getOperand(0).getType().isInteger(1)))
      return emitPermanentReject(op, "vector/per-lane branch condition as CFG");

    SmallVector<Value, 8> operands;
    for (Value operand : op->getOperands()) {
      Value mapped = lookup(operand);
      if (!mapped)
        return emitStagedDiagnostic(op,
                                    "CF operand not available in value map");
      operands.push_back(mapped);
    }

    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands(operands);
    for (Block *successor : op->getSuccessors()) {
      auto it = blockMap.find(successor);
      if (it == blockMap.end())
        return emitInternalError(op,
                                 "CF successor block missing from block map");
      state.addSuccessors(it->second);
    }
    if (failed(copyValueSafeAttrs(op, state)))
      return failure();
    builder.create(state);
    return finishLowering(op, LoweringOutcome::LoweredZeroResult);
  }

  FailureOr<Value> ensureTransferIndex(Value offset, Operation *user) {
    const SmallVector<Value, 4> *terms =
        planner.lookupContiguousOffsetScalarTerms(offset);
    if (!terms)
      return emitStagedDiagnostic(user, "non-canonical pointer offset");

    Value transferIndex;
    for (Value term : *terms) {
      Value mapped = lookup(term);
      if (!mapped)
        return emitStagedDiagnostic(user, "pointer scalar base not available");
      if (mapped.getType().isSignlessInteger(32))
        mapped = builder.create<arith::IndexCastOp>(
            user->getLoc(), builder.getIndexType(), mapped);
      else if (!mapped.getType().isIndex())
        return emitStagedDiagnostic(user,
                                    "pointer scalar base is not i32/index");
      transferIndex = transferIndex
                          ? builder
                                .create<arith::AddIOp>(user->getLoc(),
                                                       transferIndex, mapped)
                                .getResult()
                          : mapped;
    }

    if (!transferIndex)
      transferIndex = builder.create<arith::ConstantIndexOp>(user->getLoc(), 0);
    return transferIndex;
  }

  FailureOr<Value> lookupValue(Value source, Operation *user,
                               StringRef diagnostic) {
    Value mapped = lookup(source);
    if (!mapped)
      return emitStagedDiagnostic(user, diagnostic);
    return mapped;
  }

  FailureOr<Value> ensureIndexValue(Value source, Operation *user,
                                    StringRef diagnostic) {
    FailureOr<Value> mapped = lookupValue(source, user, diagnostic);
    if (failed(mapped))
      return failure();
    if ((*mapped).getType().isIndex())
      return *mapped;
    if ((*mapped).getType().isSignlessInteger(32))
      return builder
          .create<arith::IndexCastOp>(user->getLoc(), builder.getIndexType(),
                                      *mapped)
          .getResult();
    return emitStagedDiagnostic(user, "SCF loop bound is not i32/index");
  }

  FailureOr<Value> ensureTailMask(Value mask, Operation *user,
                                  StringRef sparseDiagnostic) {
    if (mask != planner.getCommonPlan().tailMaskValue)
      return emitStagedDiagnostic(user, sparseDiagnostic);

    Operation *cmp = mask.getDefiningOp();
    auto cmpi = llvm::dyn_cast_or_null<arith::CmpIOp>(cmp);
    if (!cmpi)
      return emitStagedDiagnostic(user, sparseDiagnostic);
    FailureOr<Value> index = ensureTransferIndex(cmpi.getOperand(0), user);
    if (failed(index))
      return failure();
    Value n = lookup(planner.getCommonPlan().sizeArg);
    if (!n)
      return emitInternalError(user, "tail size argument not mapped");
    Value remaining = builder.create<arith::SubIOp>(user->getLoc(), n, *index);
    return createVectorCreateMask(builder, user->getLoc(), remaining);
  }

  FailureOr<Value> ensureMemoryMask(Value mask, Operation *user,
                                    StringRef sparseDiagnostic) {
    if (mask == planner.getCommonPlan().staticFullMaskValue)
      return Value();
    return ensureTailMask(mask, user, sparseDiagnostic);
  }

  Value createZeroPadding(Location loc, Type elementType) {
    if (elementType.isF16())
      return builder.create<arith::ConstantOp>(
          loc, elementType, builder.getFloatAttr(elementType, 0.0));
    if (elementType.isF32())
      return builder.create<arith::ConstantOp>(
          loc, elementType, builder.getFloatAttr(elementType, 0.0));
    if (elementType.isSignlessInteger(32))
      return builder.create<arith::ConstantOp>(
          loc, elementType, builder.getIntegerAttr(elementType, 0));
    return {};
  }

  TTIRArgInfo *lookupSourceArg(BlockArgument arg) {
    return planner.lookupArg(arg);
  }

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
  DenseMap<BlockArgument, Type> regionArgTypeOverrides;
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

  InFlightDiagnostic diag =
      inputModule.emitError()
      << "TTIR module attributes are not currently lowerable by the VC4 TTIR "
         "target profile; unknown module attrs are staged until a value-safe "
         "module attribute policy is implemented. READY_FOR_TRITON remains NO";
  for (NamedAttribute attr : inputModule->getAttrs())
    diag << "\n  attr: " << attr.getName();
  return failure();
}

static bool isValueOutputScalarType(Type type) {
  return type.isIndex() || type.isInteger(1) || type.isSignlessInteger(8) ||
         type.isSignlessInteger(16) || type.isSignlessInteger(32) ||
         type.isF16() || type.isF32();
}

static bool isValueOutputType(Type type) {
  if (isValueOutputScalarType(type))
    return true;
  if (llvm::isa<mlir::triton::PointerType>(type))
    return false;
  if (llvm::isa<TensorType>(type))
    return false;
  if (auto vector = llvm::dyn_cast<VectorType>(type))
    return vector.hasStaticShape() &&
           isValueOutputScalarType(vector.getElementType());
  if (auto memref = llvm::dyn_cast<MemRefType>(type))
    return memref.hasRank() && isValueOutputScalarType(memref.getElementType());
  return false;
}

static LogicalResult emitForbiddenValueOutputOp(Operation *op,
                                                StringRef detail = {}) {
  InFlightDiagnostic diag =
      op->emitOpError() << "forbidden operation in TTIR-to-VC4Value output: "
                        << op->getName().getStringRef() << " dialect '"
                        << op->getName().getDialectNamespace() << "'";
  if (!detail.empty())
    diag << " (" << detail << ")";
  return failure();
}

static LogicalResult verifyValueOutputTypes(Operation *op) {
  for (Type type : op->getOperandTypes()) {
    if (!isValueOutputType(type))
      return emitForbiddenValueOutputOp(
          op, "operand type is outside the value surface");
  }
  for (Type type : op->getResultTypes()) {
    if (!isValueOutputType(type))
      return emitForbiddenValueOutputOp(
          op, "result type is outside the value surface");
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument arg : block.getArguments()) {
        if (!isValueOutputType(arg.getType()))
          return emitForbiddenValueOutputOp(
              op, "block argument type is outside the value surface");
      }
    }
  }
  if (auto func = llvm::dyn_cast<FunctionOpInterface>(op)) {
    auto type = llvm::dyn_cast<FunctionType>(func.getFunctionType());
    if (!type)
      return emitForbiddenValueOutputOp(
          op, "function signature is not a builtin function type");
    for (Type input : type.getInputs()) {
      if (!isValueOutputType(input))
        return emitForbiddenValueOutputOp(
            op, "function argument type is outside the value surface");
    }
    for (Type result : type.getResults()) {
      if (!isValueOutputType(result))
        return emitForbiddenValueOutputOp(
            op, "function result type is outside the value surface");
    }
  }
  return success();
}

static LogicalResult verifyValueOutputModule(ModuleOp outputModule) {
  for (Operation &op : outputModule.getBody()->getOperations()) {
    if (!llvm::isa<func::FuncOp>(&op))
      return emitForbiddenValueOutputOp(
          &op, "top-level value output operation must be func.func");

    if (!op.hasAttr(kVC4ValueKernelAttr))
      return op.emitOpError()
             << "value output function missing vc4value.kernel";
    if (!op.hasAttr(kVC4ValueGridRankAttr))
      return op.emitOpError()
             << "value output function missing vc4value.grid_rank";
  }

  bool sawIllegal = false;
  outputModule.walk([&](Operation *op) {
    if (llvm::isa<ModuleOp>(op))
      return WalkResult::advance();

    StringRef dialect = op->getName().getDialectNamespace();
    if (hasName(op, "builtin.unrealized_conversion_cast")) {
      (void)emitForbiddenValueOutputOp(op);
      sawIllegal = true;
      return WalkResult::interrupt();
    }
    if (dialect == "builtin") {
      (void)emitForbiddenValueOutputOp(
          op, "only builtin.module is allowed as a value-output container");
      sawIllegal = true;
      return WalkResult::interrupt();
    }
    if (dialect == "func" && !llvm::isa<func::FuncOp, func::ReturnOp>(op)) {
      (void)emitForbiddenValueOutputOp(
          op, "only func.func and func.return are allowed");
      sawIllegal = true;
      return WalkResult::interrupt();
    }
    if (isForbiddenValueOutputDialect(dialect)) {
      (void)emitForbiddenValueOutputOp(op);
      sawIllegal = true;
      return WalkResult::interrupt();
    }
    if (dialect != "func" && !isAllowedValueOutputBodyDialect(dialect)) {
      (void)emitForbiddenValueOutputOp(op);
      sawIllegal = true;
      return WalkResult::interrupt();
    }
    if (failed(verifyValueOutputTypes(op))) {
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

  ConvertTritonToVC4ValuePass() = default;
  ConvertTritonToVC4ValuePass(const ConvertTritonToVC4ValuePass &other)
      : PassWrapper(other) {
    testValueOutputBoundary = other.testValueOutputBoundary;
  }

  Option<bool> testValueOutputBoundary{
      *this, "test-value-output-boundary",
      llvm::cl::desc(
          "Test-only: verify the input module as TTIR-to-VC4Value output"),
      llvm::cl::init(false)};

  StringRef getArgument() const final { return "convert-triton-to-vc4-value"; }

  StringRef getDescription() const final {
    return "Lower the locked real TTIR target-profile subset to the VC4 value "
           "surface";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
                math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
                vector::VectorDialect, mlir::vc4value::VC4ValueDialect>();
  }

  void runOnOperation() final {
    ModuleOp inputModule = getOperation();
    if (testValueOutputBoundary) {
      if (failed(verifyValueOutputModule(inputModule)))
        signalPassFailure();
      return;
    }

    SmallVector<Operation *, 4> ttFuncs;
    SmallVector<Operation *, 4> publicTTFuncs;
    SmallVector<Operation *, 4> illegalTopLevelOps;

    if (failed(validateInputModuleAttrs(inputModule))) {
      signalPassFailure();
      return;
    }

    for (Operation &op : inputModule.getBody()->getOperations()) {
      if (hasName(&op, kTTFuncOpName)) {
        ttFuncs.push_back(&op);
        if (isPublicTTFunc(&op))
          publicTTFuncs.push_back(&op);
        continue;
      }
      // Non-operation location aliases are not module body ops.  Any real
      // top-level op beside tt.func would survive into value IR and must be
      // classified explicitly by a later phase.
      illegalTopLevelOps.push_back(&op);
    }

    if (ttFuncs.empty() || publicTTFuncs.empty()) {
      inputModule.emitError(
          "expected at least one public tt.func; C++ TTIR-to-VC4Value "
          "lowering requires the locked VC4 TTIR target-profile subset. "
          "READY_FOR_TRITON remains NO");
      signalPassFailure();
      return;
    }
    if (!illegalTopLevelOps.empty()) {
      illegalTopLevelOps.front()->emitOpError()
          << "top-level operation beside tt.func is not currently lowerable "
          << "by the VC4 TTIR target profile; "
          << "READY_FOR_TRITON remains NO";
      signalPassFailure();
      return;
    }

    OwningOpRef<ModuleOp> outputModule = ModuleOp::create(inputModule.getLoc());

    {
      SmallVector<std::unique_ptr<FunctionPlanner>, 4> planners;
      std::set<std::string> valueFunctionNames;

      for (Operation *ttFunc : publicTTFuncs) {
        auto planner = std::make_unique<FunctionPlanner>(ttFunc);
        if (failed(planner->analyze())) {
          signalPassFailure();
          return;
        }

        auto fn = llvm::dyn_cast<FunctionOpInterface>(ttFunc);
        if (!fn) {
          ttFunc->emitOpError(
              "expected tt.func to implement FunctionOpInterface");
          signalPassFailure();
          return;
        }
        std::string valueName = sanitizeSymbolName(fn.getName());
        if (!valueFunctionNames.insert(valueName).second) {
          ttFunc->emitOpError()
              << "duplicate value output symbol '" << valueName
              << "' after TTIR function name sanitization is not currently "
                 "lowerable by the VC4 TTIR target profile; "
                 "READY_FOR_TRITON remains NO";
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
