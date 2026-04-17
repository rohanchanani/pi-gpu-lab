#include "vc4/Conversion/GPUToVC4/GPUToVC4.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "vc4/Dialect/VC4/VC4Ops.h"

namespace mlir::vc4 {
namespace {

struct SupportedKernelInfo {
  gpu::GPUFuncOp func;
  gpu::GlobalIdOp globalId;
  unsigned upperBoundArgIndex = 0;
  SmallVector<Operation *> logicalBodyOps;
};

static bool isDynamicF32MemRef(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.isDynamicDim(0) &&
         memref.getElementType().isF32();
}

static bool isSupportedKernelArgumentType(Type type) {
  return isDynamicF32MemRef(type) || type.isF32() || isa<IndexType>(type);
}

static bool isSupportedScalarValue(Value value) {
  return value.getType().isF32();
}

static bool isSupportedStraightLineOp(Operation *op) {
  return isa<memref::LoadOp, arith::MulFOp, arith::AddFOp, memref::StoreOp>(op);
}

static Type getLoweredUniformType(Type sourceType, Type vec16F32Type) {
  if (sourceType.isF32())
    return vec16F32Type;
  return sourceType;
}

static FailureOr<unsigned> getFunctionArgumentIndex(Value value,
                                                    gpu::GPUFuncOp func) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return func.emitError("expected value defined by a kernel argument"),
           failure();
  if (blockArg.getOwner() != &func.getBody().front())
    return func.emitError("expected value defined by the kernel entry block"),
           failure();
  return blockArg.getArgNumber();
}

static LogicalResult validateStraightLineBody(gpu::GPUFuncOp func,
                                             Value globalId,
                                             ArrayRef<Operation *> bodyOps) {
  llvm::DenseSet<Value> availableValues;
  for (BlockArgument arg : func.getBody().front().getArguments())
    availableValues.insert(arg);

  for (Operation *op : bodyOps) {
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      if (!isDynamicF32MemRef(load.getMemRef().getType()))
        return load.emitOpError(
            "requires source memref<?xf32> kernel arguments in the supported subset");
      if (load.getIndices().size() != 1 || load.getIndices().front() != globalId)
        return load.emitOpError(
            "requires scalar loads at the logical thread index in the supported subset");
      if (!load.getType().isF32())
        return load.emitOpError("requires f32 scalar loads in the supported subset");
      if (failed(getFunctionArgumentIndex(load.getMemRef(), func)))
        return failure();
      availableValues.insert(load.getResult());
      continue;
    }

    if (auto mul = dyn_cast<arith::MulFOp>(op)) {
      if (!isSupportedScalarValue(mul.getLhs()) ||
          !isSupportedScalarValue(mul.getRhs()) || !mul.getType().isF32())
        return mul.emitOpError(
            "requires f32 operands and result in the supported subset");
      if (!availableValues.contains(mul.getLhs()) ||
          !availableValues.contains(mul.getRhs()))
        return mul.emitOpError(
            "requires operands defined by kernel arguments or earlier supported ops");
      availableValues.insert(mul.getResult());
      continue;
    }

    if (auto add = dyn_cast<arith::AddFOp>(op)) {
      if (!isSupportedScalarValue(add.getLhs()) ||
          !isSupportedScalarValue(add.getRhs()) || !add.getType().isF32())
        return add.emitOpError(
            "requires f32 operands and result in the supported subset");
      if (!availableValues.contains(add.getLhs()) ||
          !availableValues.contains(add.getRhs()))
        return add.emitOpError(
            "requires operands defined by kernel arguments or earlier supported ops");
      availableValues.insert(add.getResult());
      continue;
    }

    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (!isDynamicF32MemRef(store.getMemRef().getType()))
        return store.emitOpError(
            "requires destination memref<?xf32> kernel arguments in the supported subset");
      if (store.getIndices().size() != 1 || store.getIndices().front() != globalId)
        return store.emitOpError(
            "requires scalar stores at the logical thread index in the supported subset");
      if (!store.getValue().getType().isF32())
        return store.emitOpError("requires f32 scalar stores in the supported subset");
      if (!availableValues.contains(store.getValue()))
        return store.emitOpError(
            "requires stored values defined by kernel arguments or earlier supported ops");
      if (failed(getFunctionArgumentIndex(store.getMemRef(), func)))
        return failure();
      continue;
    }

    if (op->getName().getStringRef() == "gpu.barrier")
      return op->emitOpError(
          "barriers and inter-thread synchronization are outside the currently supported gpu-to-vc4 subset");
    if (isa<arith::CmpIOp>(op))
      return op->emitOpError(
          "control-flow predicates and internal control flow are outside the currently supported gpu-to-vc4 straight-line subset");
    if (isa<scf::IfOp, scf::ForOp, scf::WhileOp>(op))
      return op->emitOpError(
          "internal control flow is outside the currently supported gpu-to-vc4 straight-line subset");

    return op->emitOpError(
        "uses an operation outside the currently supported gpu-to-vc4 straight-line subset");
  }

  return success();
}

static FailureOr<SupportedKernelInfo>
validateAndCollectSupportedKernel(gpu::GPUFuncOp func) {
  if (!func->hasAttr("gpu.kernel"))
    return func.emitError("expected gpu.func with the kernel attribute"),
           failure();

  for (Type argType : func.getArgumentTypes()) {
    if (!isSupportedKernelArgumentType(argType))
      return func.emitError(
                 "supported kernel argument types are memref<?xf32>, f32, and index"),
             failure();
  }

  if (!llvm::hasSingleElement(func.getBody()))
    return func.emitError("expected a single entry block in the supported subset"),
           failure();

  Block &entry = func.getBody().front();
  if (!isa<gpu::ReturnOp>(entry.getTerminator()))
    return func.emitError("expected gpu.return terminator"), failure();

  gpu::GlobalIdOp globalId;
  SmallVector<Operation *> nonGlobalIdEntryOps;
  for (Operation &op : entry.without_terminator()) {
    if (auto gid = dyn_cast<gpu::GlobalIdOp>(&op)) {
      if (gid.getDimension() != gpu::Dimension::x)
        return gid.emitOpError("only the x dimension is currently supported"),
               failure();
      if (globalId)
        return gid.emitOpError("expected exactly one gpu.global_id in the supported subset"),
               failure();
      globalId = gid;
      continue;
    }
    nonGlobalIdEntryOps.push_back(&op);
  }

  if (!globalId)
    return func.emitError("expected one gpu.global_id x in the kernel entry block"),
           failure();

  SupportedKernelInfo info;
  info.func = func;
  info.globalId = globalId;

  SmallVector<arith::CmpIOp> cmpOps;
  SmallVector<scf::IfOp> ifOps;
  SmallVector<Operation *> straightLineEntryOps;
  for (Operation *op : nonGlobalIdEntryOps) {
    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      cmpOps.push_back(cmp);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      ifOps.push_back(ifOp);
      continue;
    }
    if (isSupportedStraightLineOp(op)) {
      straightLineEntryOps.push_back(op);
      continue;
    }
    if (isa<gpu::BarrierOp>(op))
      return op->emitOpError(
          "barriers and inter-thread synchronization are outside the currently supported gpu-to-vc4 subset"),
             failure();
    return op->emitOpError(
               "kernel entry uses an unsupported feature; supported kernels are straight-line and may use at most one canonical in-bounds scf.if guard"),
           failure();
  }

  if (!ifOps.empty() && !straightLineEntryOps.empty())
    return func.emitError(
               "supported kernels must be either straight-line or wrapped in one canonical in-bounds guard; mixing top-level body ops with a guard is unsupported"),
           failure();

  if (ifOps.size() > 1)
    return func.emitError(
               "multiple scf.if regions are outside the currently supported gpu-to-vc4 straight-line subset"),
           failure();

  if (ifOps.empty()) {
    if (!cmpOps.empty())
      return func.emitError(
                 "top-level compare operations are only supported as part of the canonical in-bounds guard"),
             failure();
    return func.emitError(
               "unguarded kernels are not yet supported because the logical upper bound is not explicit to the backend"),
           failure();
  } else {
    scf::IfOp ifOp = ifOps.front();
    if (!cmpOps.empty() && (cmpOps.size() != 1 || cmpOps.front().getResult() != ifOp.getCondition()))
      return func.emitError(
                 "the only supported top-level compare is the canonical in-bounds guard compare feeding scf.if"),
             failure();
    if (!ifOp.getElseRegion().empty())
      return ifOp.emitOpError(
                 "else regions are outside the currently supported gpu-to-vc4 straight-line subset"),
             failure();
    if (!llvm::hasSingleElement(ifOp.getThenRegion()))
      return ifOp.emitOpError(
                 "supported in-bounds guards require a single then block"),
             failure();

    auto cmp = dyn_cast<arith::CmpIOp>(ifOp.getCondition().getDefiningOp());
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::ult ||
        cmp.getLhs() != globalId.getResult())
      return ifOp.emitOpError(
                 "supported guards must be canonical in-bounds checks of the form arith.cmpi ult %gid, %upperBound"),
             failure();
    if (!isa<BlockArgument>(cmp.getRhs()) || !isa<IndexType>(cmp.getRhs().getType()))
      return cmp.emitOpError(
                 "supported in-bounds guards require the upper bound to be an index kernel argument"),
             failure();

    FailureOr<unsigned> upperBoundArgIndex =
        getFunctionArgumentIndex(cmp.getRhs(), func);
    if (failed(upperBoundArgIndex))
      return failure();
    info.upperBoundArgIndex = *upperBoundArgIndex;

    Block &thenBlock = ifOp.getThenRegion().front();
    for (Operation &op : thenBlock.without_terminator())
      info.logicalBodyOps.push_back(&op);
  }

  if (failed(validateStraightLineBody(func, globalId.getResult(),
                                      info.logicalBodyOps)))
    return failure();

  return info;
}

static void lowerSupportedKernel(const SupportedKernelInfo &info) {
  gpu::GPUFuncOp func = info.func;
  Location loc = func.getLoc();
  MLIRContext *ctx = func.getContext();
  Region &bodyRegion = func.getBody();
  Block &oldEntry = bodyRegion.front();

  SmallVector<Type> argTypes(oldEntry.getArgumentTypes().begin(),
                             oldEntry.getArgumentTypes().end());
  auto *newEntry = new Block();
  for (Type argType : argTypes)
    newEntry->addArgument(argType, loc);
  bodyRegion.push_back(newEntry);

  ImplicitLocOpBuilder builder(loc, ctx);
  builder.setInsertionPointToStart(newEntry);

  Type indexType = builder.getIndexType();
  Type vec16F32Type = VectorType::get({16}, builder.getF32Type());

  auto qpu = GetBuiltinOp::create(builder, indexType,
                                  builder.getStringAttr("qpu_id"));
  auto numQpus = GetBuiltinOp::create(builder, indexType,
                                      builder.getStringAttr("num_qpus"));

  llvm::SmallDenseMap<Value, Value> valueMap;
  for (auto [argIndex, arg] : llvm::enumerate(oldEntry.getArguments())) {
    Type loweredType = getLoweredUniformType(arg.getType(), vec16F32Type);
    auto uniform = GetUniformOp::create(
        builder, loweredType,
        builder.getI32IntegerAttr(static_cast<int32_t>(argIndex)));
    valueMap[arg] = uniform.getResult();
  }

  Value upperBound =
      valueMap.lookup(oldEntry.getArgument(info.upperBoundArgIndex));

  auto c16 = arith::ConstantIndexOp::create(builder, 16);
  auto base = arith::MulIOp::create(builder, qpu.getResult(), c16);
  auto stride = arith::MulIOp::create(builder, numQpus.getResult(), c16);

  auto loop = scf::ForOp::create(builder, base, upperBound, stride);
  builder.setInsertionPointToStart(loop.getBody());
  Value iv = loop.getInductionVar();

  unsigned nextStagingSlot = 0;
  for (Operation *op : info.logicalBodyOps) {
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      unsigned slot = nextStagingSlot++;
      DmaLoadOp::create(builder, valueMap.lookup(load.getMemRef()), iv,
                        builder.getI32IntegerAttr(static_cast<int32_t>(slot)));
      auto staged = StagedReadOp::create(
          builder, vec16F32Type,
          builder.getI32IntegerAttr(static_cast<int32_t>(slot)));
      valueMap[load.getResult()] = staged.getResult();
      continue;
    }

    if (auto mul = dyn_cast<arith::MulFOp>(op)) {
      auto loweredMul = FMulOp::create(builder, vec16F32Type,
                                       valueMap.lookup(mul.getLhs()),
                                       valueMap.lookup(mul.getRhs()));
      valueMap[mul.getResult()] = loweredMul.getResult();
      continue;
    }

    if (auto add = dyn_cast<arith::AddFOp>(op)) {
      auto loweredAdd = FAddOp::create(builder, vec16F32Type,
                                       valueMap.lookup(add.getLhs()),
                                       valueMap.lookup(add.getRhs()));
      valueMap[add.getResult()] = loweredAdd.getResult();
      continue;
    }

    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      unsigned slot = nextStagingSlot++;
      StagedWriteOp::create(builder,
                            builder.getI32IntegerAttr(static_cast<int32_t>(slot)),
                            valueMap.lookup(store.getValue()));
      DmaStoreOp::create(builder,
                         builder.getI32IntegerAttr(static_cast<int32_t>(slot)),
                         valueMap.lookup(store.getMemRef()), iv);
      continue;
    }

    llvm_unreachable("validated kernel body contains an unsupported op");
  }

  builder.setInsertionPointToEnd(newEntry);
  gpu::ReturnOp::create(builder);

  oldEntry.erase();
}

struct LowerGPUToVC4Pass
    : public PassWrapper<LowerGPUToVC4Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGPUToVC4Pass)

  StringRef getArgument() const final { return "lower-gpu-to-vc4"; }
  StringRef getDescription() const final {
    return "Lower the supported straight-line gpu.func subset to VC4 persistent-worker form";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect,
                    scf::SCFDialect, VC4Dialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<gpu::GPUFuncOp> kernelFuncs;
    module.walk([&](gpu::GPUFuncOp func) {
      if (func->hasAttr("gpu.kernel"))
        kernelFuncs.push_back(func);
    });

    if (kernelFuncs.empty())
      return;

    for (gpu::GPUFuncOp func : kernelFuncs) {
      FailureOr<SupportedKernelInfo> info = validateAndCollectSupportedKernel(func);
      if (failed(info)) {
        signalPassFailure();
        return;
      }
      lowerSupportedKernel(*info);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLowerGPUToVC4Pass() {
  return std::make_unique<LowerGPUToVC4Pass>();
}

// Deprecated compatibility shim while downstream callers migrate to the generic
// pass name.
std::unique_ptr<Pass> createLowerGPUSaxpyToVC4Pass() {
  return createLowerGPUToVC4Pass();
}

void registerGPUToVC4Passes() { PassRegistration<LowerGPUToVC4Pass>(); }

} // namespace mlir::vc4
