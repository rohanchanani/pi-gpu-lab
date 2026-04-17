#include "vc4/Target/KernelModel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "vc4/Dialect/VC4/VC4Dialect.h"

namespace mlir::vc4 {
namespace {

static bool isDynamicF32MemRef(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.isDynamicDim(0) &&
         memref.getElementType().isF32();
}

static Type getExpectedLoweredUniformType(Type sourceType, MLIRContext *ctx) {
  if (sourceType.isF32())
    return VectorType::get({16}, Float32Type::get(ctx));
  return sourceType;
}

static std::optional<int64_t> getConstantIndexValue(Value value) {
  auto constant = dyn_cast_or_null<arith::ConstantIndexOp>(value.getDefiningOp());
  if (!constant)
    return std::nullopt;
  return constant.value();
}

static FailureOr<gpu::GPUFuncOp> extractSingleKernel(ModuleOp module) {
  SmallVector<gpu::GPUModuleOp> gpuModules;
  module.walk([&](gpu::GPUModuleOp gpuModule) { gpuModules.push_back(gpuModule); });
  if (gpuModules.size() != 1) {
    module.emitError(
        "vc4 kernel model extraction expects exactly one gpu.module");
    return failure();
  }

  SmallVector<gpu::GPUFuncOp> gpuFuncs;
  gpuModules.front().walk([&](gpu::GPUFuncOp func) {
    if (func->hasAttr("gpu.kernel"))
      gpuFuncs.push_back(func);
  });
  if (gpuFuncs.size() != 1) {
    gpuModules.front().emitError(
        "vc4 kernel model extraction expects exactly one gpu.func with the kernel attribute");
    return failure();
  }

  return gpuFuncs.front();
}

static std::string formatValueRef(const KernelValueRef &value) {
  switch (value.kind) {
  case KernelValueKind::UniformSlot:
    return (llvm::Twine("uniform") + llvm::Twine(value.index)).str();
  case KernelValueKind::Builtin:
    return (llvm::Twine("builtin(") + stringifyBuiltinKind(value.builtinKind) + ")")
        .str();
  case KernelValueKind::LoopIndex:
    return "iv";
  case KernelValueKind::Temporary:
    return (llvm::Twine("t") + llvm::Twine(value.index)).str();
  }
  llvm_unreachable("unknown kernel value kind");
}

template <typename OpTy, typename Predicate>
static FailureOr<OpTy> findUniqueMatchingOp(Block &block,
                                            llvm::SmallPtrSetImpl<Operation *> &matched,
                                            Predicate predicate,
                                            StringRef errorMessage) {
  OpTy result;
  for (Operation &op : block.without_terminator()) {
    auto candidate = dyn_cast<OpTy>(&op);
    if (!candidate || !predicate(candidate))
      continue;
    if (result)
      return block.getParentOp()->emitError(errorMessage), failure();
    result = candidate;
    matched.insert(&op);
  }
  if (!result)
    return block.getParentOp()->emitError(errorMessage), failure();
  return result;
}

template <typename OpTy>
static SmallVector<OpTy> collectOps(Block &block) {
  SmallVector<OpTy> results;
  for (Operation &op : block.without_terminator()) {
    if (auto candidate = dyn_cast<OpTy>(&op))
      results.push_back(candidate);
  }
  return results;
}

static LogicalResult rejectUnexpectedOps(
    Block &block, llvm::SmallPtrSetImpl<Operation *> &matched,
    llvm::function_ref<bool(Operation &)> isAllowedKind, StringRef errorMessage) {
  for (Operation &op : block.without_terminator()) {
    if (!isAllowedKind(op))
      return op.emitError(errorMessage);
    if (!matched.contains(&op))
      return op.emitError(errorMessage);
  }
  return success();
}

static FailureOr<unsigned> getScaledBuiltinMultiplier(
    Value value, BuiltinKind builtinKind, llvm::SmallPtrSetImpl<Operation *> &matchedEntryOps,
    StringRef errorMessage) {
  auto mul = dyn_cast_or_null<arith::MulIOp>(value.getDefiningOp());
  if (!mul)
    return emitError(value.getLoc(), errorMessage), failure();

  auto lhsBuiltin = dyn_cast_or_null<GetBuiltinOp>(mul.getLhs().getDefiningOp());
  auto rhsBuiltin = dyn_cast_or_null<GetBuiltinOp>(mul.getRhs().getDefiningOp());
  std::optional<int64_t> lhsConst = getConstantIndexValue(mul.getLhs());
  std::optional<int64_t> rhsConst = getConstantIndexValue(mul.getRhs());

  auto matchBuiltinAndConstant =
      [&](GetBuiltinOp builtinOp, std::optional<int64_t> constant) -> FailureOr<unsigned> {
    if (!builtinOp || builtinOp.getBuiltinKind() != builtinKind || !constant ||
        *constant <= 0)
      return emitError(value.getLoc(), errorMessage), failure();
    matchedEntryOps.insert(mul.getOperation());
    return static_cast<unsigned>(*constant);
  };

  if (lhsBuiltin && rhsConst)
    return matchBuiltinAndConstant(lhsBuiltin, rhsConst);
  if (rhsBuiltin && lhsConst)
    return matchBuiltinAndConstant(rhsBuiltin, lhsConst);
  return emitError(value.getLoc(), errorMessage), failure();
}

static FailureOr<KernelValueRef> lookupValueRef(
    Value value, llvm::SmallDenseMap<Value, KernelValueRef> &valueRefs,
    Operation *op, StringRef errorMessage) {
  auto it = valueRefs.find(value);
  if (it == valueRefs.end())
    return op->emitOpError(errorMessage), failure();
  return it->second;
}

static FailureOr<llvm::SmallVector<BodyOperationInfo>> extractBodyOperations(
    Block &body, scf::ForOp loop,
    llvm::SmallDenseMap<Value, GetUniformOp> &uniformsByValue,
    llvm::SmallDenseMap<Value, KernelValueRef> &valueRefs) {
  llvm::SmallVector<BodyOperationInfo> bodyOps;
  llvm::SmallPtrSet<Operation *, 16> matchedBodyOps;
  unsigned nextTemporaryIndex = 0;

  for (Operation &op : body.without_terminator()) {
    if (auto dmaLoad = dyn_cast<DmaLoadOp>(&op)) {
      auto sourceUniform = uniformsByValue.find(dmaLoad.getSource());
      if (sourceUniform == uniformsByValue.end())
        return dmaLoad.emitOpError(
                   "expected dma_load source to come from vc4.get_uniform in the supported lowered subset"),
               failure();
      if (!isDynamicF32MemRef(sourceUniform->second.getType()))
        return dmaLoad.emitOpError(
                   "expected dma_load source uniform to be memref<?xf32>"),
               failure();
      if (dmaLoad.getGlobalOffset() != loop.getInductionVar())
        return dmaLoad.emitOpError(
                   "expected dma_load offset to use the persistent-worker induction variable"),
               failure();

      matchedBodyOps.insert(&op);
      bodyOps.push_back(BodyOperationInfo{
          BodyOperationKind::DmaLoad,
          sourceUniform->second.getIndex(),
          dmaLoad.getSlot(),
          KernelValueRef::loopIndex(),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt});
      continue;
    }

    if (auto stagedRead = dyn_cast<StagedReadOp>(&op)) {
      matchedBodyOps.insert(&op);
      KernelValueRef resultRef = KernelValueRef::temporary(nextTemporaryIndex++);
      valueRefs[stagedRead.getResult()] = resultRef;
      bodyOps.push_back(BodyOperationInfo{
          BodyOperationKind::StagedRead,
          std::nullopt,
          stagedRead.getSlot(),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          resultRef});
      continue;
    }

    if (auto mul = dyn_cast<FMulOp>(&op)) {
      FailureOr<KernelValueRef> lhs = lookupValueRef(
          mul.getLhs(), valueRefs, &op,
          "expected fmul operands defined by earlier lowered values");
      FailureOr<KernelValueRef> rhs = lookupValueRef(
          mul.getRhs(), valueRefs, &op,
          "expected fmul operands defined by earlier lowered values");
      if (failed(lhs) || failed(rhs))
        return failure();

      matchedBodyOps.insert(&op);
      KernelValueRef resultRef = KernelValueRef::temporary(nextTemporaryIndex++);
      valueRefs[mul.getResult()] = resultRef;
      bodyOps.push_back(BodyOperationInfo{
          BodyOperationKind::FMul,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          *lhs,
          *rhs,
          resultRef});
      continue;
    }

    if (auto add = dyn_cast<FAddOp>(&op)) {
      FailureOr<KernelValueRef> lhs = lookupValueRef(
          add.getLhs(), valueRefs, &op,
          "expected fadd operands defined by earlier lowered values");
      FailureOr<KernelValueRef> rhs = lookupValueRef(
          add.getRhs(), valueRefs, &op,
          "expected fadd operands defined by earlier lowered values");
      if (failed(lhs) || failed(rhs))
        return failure();

      matchedBodyOps.insert(&op);
      KernelValueRef resultRef = KernelValueRef::temporary(nextTemporaryIndex++);
      valueRefs[add.getResult()] = resultRef;
      bodyOps.push_back(BodyOperationInfo{
          BodyOperationKind::FAdd,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          *lhs,
          *rhs,
          resultRef});
      continue;
    }

    if (auto stagedWrite = dyn_cast<StagedWriteOp>(&op)) {
      FailureOr<KernelValueRef> source = lookupValueRef(
          stagedWrite.getValue(), valueRefs, &op,
          "expected staged_write source defined by earlier lowered values");
      if (failed(source))
        return failure();

      matchedBodyOps.insert(&op);
      bodyOps.push_back(BodyOperationInfo{
          BodyOperationKind::StagedWrite,
          std::nullopt,
          stagedWrite.getSlot(),
          std::nullopt,
          *source,
          std::nullopt,
          std::nullopt,
          std::nullopt});
      continue;
    }

    if (auto dmaStore = dyn_cast<DmaStoreOp>(&op)) {
      auto targetUniform = uniformsByValue.find(dmaStore.getTarget());
      if (targetUniform == uniformsByValue.end())
        return dmaStore.emitOpError(
                   "expected dma_store target to come from vc4.get_uniform in the supported lowered subset"),
               failure();
      if (!isDynamicF32MemRef(targetUniform->second.getType()))
        return dmaStore.emitOpError(
                   "expected dma_store target uniform to be memref<?xf32>"),
               failure();
      if (dmaStore.getGlobalOffset() != loop.getInductionVar())
        return dmaStore.emitOpError(
                   "expected dma_store offset to use the persistent-worker induction variable"),
               failure();

      matchedBodyOps.insert(&op);
      bodyOps.push_back(BodyOperationInfo{
          BodyOperationKind::DmaStore,
          targetUniform->second.getIndex(),
          dmaStore.getSlot(),
          KernelValueRef::loopIndex(),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt});
      continue;
    }

    return op.emitOpError(
               "unexpected loop-body operation outside the currently supported lowered subset"),
           failure();
  }

  if (failed(rejectUnexpectedOps(
          body, matchedBodyOps,
          [](Operation &op) {
            return isa<DmaLoadOp, StagedReadOp, FMulOp, FAddOp, StagedWriteOp,
                       DmaStoreOp>(op);
          },
          "unexpected loop-body operation outside the currently supported lowered subset")))
    return failure();

  return bodyOps;
}

} // namespace

KernelValueRef KernelValueRef::uniformSlot(unsigned slot) {
  return KernelValueRef{KernelValueKind::UniformSlot, slot, BuiltinKind::qpu_id};
}

KernelValueRef KernelValueRef::builtin(BuiltinKind builtinKind) {
  return KernelValueRef{KernelValueKind::Builtin, 0u, builtinKind};
}

KernelValueRef KernelValueRef::loopIndex() {
  return KernelValueRef{KernelValueKind::LoopIndex, 0u, BuiltinKind::qpu_id};
}

KernelValueRef KernelValueRef::temporary(unsigned index) {
  return KernelValueRef{KernelValueKind::Temporary, index, BuiltinKind::qpu_id};
}

FailureOr<LoweredKernelArtifactModel> extractLoweredKernelArtifactModel(Operation *op) {
  if (auto module = dyn_cast<ModuleOp>(op)) {
    FailureOr<gpu::GPUFuncOp> func = extractSingleKernel(module);
    if (failed(func))
      return failure();
    return extractLoweredKernelArtifactModel(*func);
  }
  if (auto func = dyn_cast<gpu::GPUFuncOp>(op))
    return extractLoweredKernelArtifactModel(func);
  return op->emitError(
             "vc4 kernel model extraction expects a builtin.module or gpu.func"),
         failure();
}

FailureOr<LoweredKernelArtifactModel>
extractLoweredKernelArtifactModel(gpu::GPUFuncOp func) {
  if (!func->hasAttr("gpu.kernel"))
    return func.emitError("expected gpu.func with the kernel attribute"), failure();

  if (!llvm::hasSingleElement(func.getBody()))
    return func.emitError("expected a single entry block"), failure();

  Block &entry = func.getBody().front();
  if (!isa<gpu::ReturnOp>(entry.getTerminator()))
    return func.emitError("expected gpu.return terminator"), failure();

  for (Type argType : func.getArgumentTypes()) {
    if (!(isDynamicF32MemRef(argType) || argType.isF32() || isa<IndexType>(argType)))
      return func.emitError(
                 "supported public argument types are memref<?xf32>, f32, and index"),
             failure();
  }

  llvm::SmallPtrSet<Operation *, 16> matchedEntryOps;

  auto qpu = findUniqueMatchingOp<GetBuiltinOp>(
      entry, matchedEntryOps,
      [&](GetBuiltinOp op) { return op.getBuiltinKind() == BuiltinKind::qpu_id; },
      "expected exactly one vc4.get_builtin qpu_id");
  if (failed(qpu))
    return failure();

  auto numQpus = findUniqueMatchingOp<GetBuiltinOp>(
      entry, matchedEntryOps,
      [&](GetBuiltinOp op) { return op.getBuiltinKind() == BuiltinKind::num_qpus; },
      "expected exactly one vc4.get_builtin num_qpus");
  if (failed(numQpus))
    return failure();

  auto uniformOps = collectOps<GetUniformOp>(entry);
  if (uniformOps.size() != func.getNumArguments())
    return func.emitError(
               "expected exactly one vc4.get_uniform per source kernel argument"),
           failure();

  llvm::SmallDenseMap<unsigned, GetUniformOp> uniformsBySlot;
  for (GetUniformOp uniform : uniformOps) {
    unsigned slot = uniform.getIndex();
    if (slot >= func.getNumArguments())
      return uniform.emitOpError("references a uniform slot outside the source signature"),
             failure();
    if (uniformsBySlot.contains(slot))
      return uniform.emitOpError("duplicates a logical uniform slot"), failure();
    Type expectedType =
        getExpectedLoweredUniformType(func.getArgument(slot).getType(), func.getContext());
    if (uniform.getType() != expectedType)
      return uniform.emitOpError()
             << "has type " << uniform.getType()
             << " but the lowered form expects " << expectedType;
    uniformsBySlot[slot] = uniform;
    matchedEntryOps.insert(uniform.getOperation());
  }

  for (arith::ConstantIndexOp constant : collectOps<arith::ConstantIndexOp>(entry))
    matchedEntryOps.insert(constant.getOperation());

  auto loop = findUniqueMatchingOp<scf::ForOp>(
      entry, matchedEntryOps,
      [&](scf::ForOp) { return true; },
      "expected exactly one persistent-worker scf.for loop");
  if (failed(loop))
    return failure();

  FailureOr<unsigned> baseMultiplier = getScaledBuiltinMultiplier(
      loop->getLowerBound(), BuiltinKind::qpu_id, matchedEntryOps,
      "expected the loop lower bound to compute qpu_id multiplied by the VC4 lane width");
  if (failed(baseMultiplier))
    return failure();

  FailureOr<unsigned> strideMultiplier = getScaledBuiltinMultiplier(
      loop->getStep(), BuiltinKind::num_qpus, matchedEntryOps,
      "expected the loop step to compute num_qpus multiplied by the VC4 lane width");
  if (failed(strideMultiplier))
    return failure();
  if (*baseMultiplier != *strideMultiplier)
    return loop->emitError(
               "expected the persistent-worker base and stride multipliers to use the same lane width"),
           failure();

  auto upperBoundUniform = dyn_cast_or_null<GetUniformOp>(
      loop->getUpperBound().getDefiningOp());
  if (!upperBoundUniform)
    return loop->emitError(
               "expected the loop upper bound to come from vc4.get_uniform in the supported lowered subset"),
           failure();
  unsigned upperBoundSlot = upperBoundUniform.getIndex();
  if (!isa<IndexType>(func.getArgument(upperBoundSlot).getType()))
    return loop->emitError(
               "expected the loop upper bound uniform to correspond to an index source argument"),
           failure();

  if (failed(rejectUnexpectedOps(
          entry, matchedEntryOps,
          [](Operation &op) {
            return isa<GetBuiltinOp, GetUniformOp, arith::ConstantIndexOp,
                       arith::MulIOp, scf::ForOp>(op);
          },
          "unexpected entry-block operation outside the currently supported lowered subset")))
    return failure();

  if (!llvm::hasSingleElement(loop->getRegion()))
    return loop->emitError("expected a single loop body block"), failure();
  Block &body = loop->getRegion().front();
  if (!isa<scf::YieldOp>(body.getTerminator()))
    return loop->emitError("expected scf.yield terminator"), failure();

  LoweredKernelArtifactModel model;
  model.kernelName = func.getName().str();
  for (auto [argIndex, argType] : llvm::enumerate(func.getArgumentTypes())) {
    model.publicArguments.push_back(
        PublicArgumentInfo{static_cast<unsigned>(argIndex), argType});
    model.uniformStream.push_back(UniformStreamEntry{
        static_cast<unsigned>(argIndex),
        getExpectedLoweredUniformType(argType, func.getContext()),
        UniformSourceKind::PublicArgument,
        static_cast<unsigned>(argIndex),
        BuiltinKind::qpu_id});
  }

  model.builtinSuffixStart = func.getNumArguments();
  model.uniformStream.push_back(UniformStreamEntry{
      static_cast<unsigned>(func.getNumArguments()),
      qpu->getType(),
      UniformSourceKind::Builtin,
      0u,
      BuiltinKind::qpu_id});
  model.uniformStream.push_back(UniformStreamEntry{
      static_cast<unsigned>(func.getNumArguments() + 1),
      numQpus->getType(),
      UniformSourceKind::Builtin,
      0u,
      BuiltinKind::num_qpus});

  model.execution.workerIdBuiltin = BuiltinKind::qpu_id;
  model.execution.workerCountBuiltin = BuiltinKind::num_qpus;
  model.execution.laneWidth = *baseMultiplier;
  model.execution.baseMultiplier = *baseMultiplier;
  model.execution.strideMultiplier = *strideMultiplier;
  model.execution.upperBoundUniformSlot = upperBoundSlot;
  model.execution.assumesUpperBoundMultipleOfLaneWidth = true;

  llvm::SmallDenseMap<Value, KernelValueRef> valueRefs;
  valueRefs[qpu->getResult()] = KernelValueRef::builtin(BuiltinKind::qpu_id);
  valueRefs[numQpus->getResult()] = KernelValueRef::builtin(BuiltinKind::num_qpus);
  llvm::SmallDenseMap<Value, GetUniformOp> uniformsByValue;
  for (auto [slot, uniform] : uniformsBySlot) {
    valueRefs[uniform.getResult()] = KernelValueRef::uniformSlot(slot);
    uniformsByValue[uniform.getResult()] = uniform;
  }
  valueRefs[loop->getInductionVar()] = KernelValueRef::loopIndex();
  FailureOr<llvm::SmallVector<BodyOperationInfo>> bodyOps =
      extractBodyOperations(body, *loop, uniformsByValue, valueRefs);
  if (failed(bodyOps))
    return failure();
  model.body = std::move(*bodyOps);

  return model;
}

LogicalResult printLoweredKernelArtifactModel(const LoweredKernelArtifactModel &model,
                                              llvm::raw_ostream &output) {
  output << "kernel @" << model.kernelName << "\n";
  output << "public_args:\n";
  for (const PublicArgumentInfo &arg : model.publicArguments) {
    output << "  arg" << arg.index << ": ";
    arg.type.print(output);
    output << "\n";
  }

  output << "uniform_stream:\n";
  for (const UniformStreamEntry &entry : model.uniformStream) {
    output << "  uniform" << entry.slot << ": ";
    if (entry.sourceKind == UniformSourceKind::PublicArgument) {
      output << "arg" << entry.publicArgumentIndex << " : ";
    } else {
      output << "builtin " << stringifyBuiltinKind(entry.builtinKind) << " : ";
    }
    entry.type.print(output);
    output << "\n";
  }

  output << "builtin_suffix_start: " << model.builtinSuffixStart << "\n";
  output << "execution:\n";
  output << "  worker_id_builtin: " << stringifyBuiltinKind(model.execution.workerIdBuiltin)
         << "\n";
  output << "  worker_count_builtin: "
         << stringifyBuiltinKind(model.execution.workerCountBuiltin) << "\n";
  output << "  lane_width: " << model.execution.laneWidth << "\n";
  output << "  base = " << stringifyBuiltinKind(model.execution.workerIdBuiltin)
         << " * " << model.execution.baseMultiplier << "\n";
  output << "  stride = " << stringifyBuiltinKind(model.execution.workerCountBuiltin)
         << " * " << model.execution.strideMultiplier << "\n";
  output << "  upper_bound: uniform" << model.execution.upperBoundUniformSlot << "\n";
  output << "  assumes_upper_bound_multiple_of_lane_width: "
         << (model.execution.assumesUpperBoundMultipleOfLaneWidth ? "true" : "false")
         << "\n";

  output << "body:\n";
  for (const BodyOperationInfo &op : model.body) {
    output << "  ";
    switch (op.kind) {
    case BodyOperationKind::DmaLoad:
      output << "dma_load uniform" << *op.memrefUniformSlot << "["
             << formatValueRef(*op.offset) << "] -> stage" << *op.stagingSlot << "\n";
      break;
    case BodyOperationKind::StagedRead:
      output << "staged_read stage" << *op.stagingSlot << " -> "
             << formatValueRef(*op.result) << "\n";
      break;
    case BodyOperationKind::FMul:
      output << "fmul " << formatValueRef(*op.lhs) << ", "
             << formatValueRef(*op.rhs) << " -> " << formatValueRef(*op.result)
             << "\n";
      break;
    case BodyOperationKind::FAdd:
      output << "fadd " << formatValueRef(*op.lhs) << ", "
             << formatValueRef(*op.rhs) << " -> " << formatValueRef(*op.result)
             << "\n";
      break;
    case BodyOperationKind::StagedWrite:
      output << "staged_write " << formatValueRef(*op.source) << " -> stage"
             << *op.stagingSlot << "\n";
      break;
    case BodyOperationKind::DmaStore:
      output << "dma_store stage" << *op.stagingSlot << " -> uniform"
             << *op.memrefUniformSlot << "[" << formatValueRef(*op.offset) << "]\n";
      break;
    }
  }

  return success();
}

void registerToVC4KernelModelTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-vc4-kernel-model",
      "Extract the lowered VC4 kernel artifact model for the currently supported straight-line subset",
      [](Operation *op, llvm::raw_ostream &output) {
        FailureOr<LoweredKernelArtifactModel> model =
            extractLoweredKernelArtifactModel(op);
        if (failed(model))
          return failure();
        return printLoweredKernelArtifactModel(*model, output);
      },
      [](DialectRegistry &registry) {
        registerAllDialects(registry);
        registry.insert<VC4Dialect>();
      });
}

} // namespace mlir::vc4
