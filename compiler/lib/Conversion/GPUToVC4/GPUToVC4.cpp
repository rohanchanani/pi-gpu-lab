#include "vc4/Conversion/GPUToVC4/GPUToVC4.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "vc4/Dialect/VC4/VC4Ops.h"

namespace mlir::vc4 {
namespace {

static bool isDynamicF32MemRef(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.isDynamicDim(0) &&
         memref.getElementType().isF32();
}

static LogicalResult validateSupportedKernel(gpu::GPUFuncOp func) {
  if (!func->hasAttr("gpu.kernel"))
    return func.emitError("expected gpu.func with the kernel attribute");
  if (func.getNumArguments() != 4)
    return func.emitError("expected exactly four kernel arguments: x, y, a, n");

  TypeRange args = func.getArgumentTypes();
  if (!isDynamicF32MemRef(args[0]) || !isDynamicF32MemRef(args[1]) ||
      !args[2].isF32() || !isa<IndexType>(args[3]))
    return func.emitError("expected argument types (memref<?xf32>, memref<?xf32>, f32, index)");

  if (!llvm::hasSingleElement(func.getBody()))
    return func.emitError("expected a single entry block");

  Block &entry = func.getBody().front();
  auto ops = entry.without_terminator();
  if (std::distance(ops.begin(), ops.end()) != 3)
    return func.emitError("expected kernel body shape: gpu.global_id, arith.cmpi, scf.if");

  Operation *first = &*ops.begin();
  Operation *second = &*std::next(ops.begin());
  Operation *third = &*std::next(ops.begin(), 2);

  auto gid = dyn_cast<gpu::GlobalIdOp>(first);
  if (!gid || gid.getDimension() != gpu::Dimension::x)
    return func.emitError("expected first op to be gpu.global_id x");

  auto cmp = dyn_cast<arith::CmpIOp>(second);
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::ult ||
      cmp.getLhs() != gid.getResult() || cmp.getRhs() != entry.getArgument(3))
    return func.emitError("expected second op to be arith.cmpi ult %gid, %n");

  auto ifOp = dyn_cast<scf::IfOp>(third);
  if (!ifOp || ifOp.getElseRegion().empty() == false ||
      ifOp.getCondition() != cmp.getResult())
    return func.emitError("expected third op to be scf.if on the in-bounds compare");

  if (!llvm::hasSingleElement(ifOp.getThenRegion()))
    return func.emitError("expected scf.if to contain a single then block");

  Block &thenBlock = ifOp.getThenRegion().front();
  auto thenOps = thenBlock.without_terminator();
  if (std::distance(thenOps.begin(), thenOps.end()) != 5)
    return func.emitError("expected then block shape: load x, load y, mulf, addf, store");

  auto xLoad = dyn_cast<memref::LoadOp>(&*thenOps.begin());
  auto yLoad = dyn_cast<memref::LoadOp>(&*std::next(thenOps.begin()));
  auto mul = dyn_cast<arith::MulFOp>(&*std::next(thenOps.begin(), 2));
  auto add = dyn_cast<arith::AddFOp>(&*std::next(thenOps.begin(), 3));
  auto store = dyn_cast<memref::StoreOp>(&*std::next(thenOps.begin(), 4));
  if (!xLoad || !yLoad || !mul || !add || !store)
    return func.emitError("expected then block ops: memref.load, memref.load, arith.mulf, arith.addf, memref.store");

  if (xLoad.getMemRef() != entry.getArgument(0) || yLoad.getMemRef() != entry.getArgument(1) ||
      xLoad.getIndices().size() != 1 || yLoad.getIndices().size() != 1 ||
      xLoad.getIndices().front() != gid.getResult() ||
      yLoad.getIndices().front() != gid.getResult())
    return func.emitError("expected scalar loads from x[%gid] and y[%gid]");

  bool mulMatches = (mul.getLhs() == entry.getArgument(2) && mul.getRhs() == xLoad.getResult()) ||
                    (mul.getRhs() == entry.getArgument(2) && mul.getLhs() == xLoad.getResult());
  if (!mulMatches)
    return func.emitError("expected multiply to be a * x");

  bool addMatches = (add.getLhs() == mul.getResult() && add.getRhs() == yLoad.getResult()) ||
                    (add.getRhs() == mul.getResult() && add.getLhs() == yLoad.getResult());
  if (!addMatches)
    return func.emitError("expected add to be (a * x) + y");

  if (store.getValue() != add.getResult() || store.getMemRef() != entry.getArgument(1) ||
      store.getIndices().size() != 1 || store.getIndices().front() != gid.getResult())
    return func.emitError("expected store to write result back to y[%gid]");

  if (!isa<gpu::ReturnOp>(entry.getTerminator()))
    return func.emitError("expected gpu.return terminator");

  return success();
}

static void lowerKernelBody(gpu::GPUFuncOp func) {
  Location loc = func.getLoc();
  MLIRContext *ctx = func.getContext();
  Region &bodyRegion = func.getBody();
  Block &oldEntry = bodyRegion.front();
  SmallVector<Type> argTypes(oldEntry.getArgumentTypes().begin(),
                             oldEntry.getArgumentTypes().end());
  SmallVector<Location> argLocs(argTypes.size(), loc);
  auto *entry = new Block();
  for (auto [argType, argLoc] : llvm::zip(argTypes, argLocs))
    entry->addArgument(argType, argLoc);
  bodyRegion.push_back(entry);
  oldEntry.erase();

  ImplicitLocOpBuilder builder(loc, ctx);
  builder.setInsertionPointToStart(entry);

  Type indexType = builder.getIndexType();
  Type vecType = VectorType::get({16}, builder.getF32Type());
  Type xType = entry->getArgument(0).getType();
  Type yType = entry->getArgument(1).getType();

  auto qpu = GetBuiltinOp::create(builder, indexType, "qpu_id");
  auto numQpus = GetBuiltinOp::create(builder, indexType, "num_qpus");

  auto x = GetUniformOp::create(builder, xType, 0u);
  auto y = GetUniformOp::create(builder, yType, 1u);
  auto a = GetUniformOp::create(builder, vecType, 2u);
  auto n = GetUniformOp::create(builder, indexType, 3u);

  auto c0 = arith::ConstantIndexOp::create(builder, 0);
  auto c1 = arith::ConstantIndexOp::create(builder, 1);
  auto c16 = arith::ConstantIndexOp::create(builder, 16);

  auto base = arith::MulIOp::create(builder, qpu.getResult(), c16);
  auto stride = arith::MulIOp::create(builder, numQpus.getResult(), c16);
  auto remainder = arith::RemUIOp::create(builder, n.getResult(), c16);
  auto isAligned =
      arith::CmpIOp::create(builder, arith::CmpIPredicate::eq, remainder, c0);
  cf::AssertOp::create(builder, isAligned,
                       "lower-gpu-saxpy-to-vc4 requires n to be a multiple of 16");

  auto loop = scf::ForOp::create(builder, base, n.getResult(), stride);
  builder.setInsertionPointToStart(loop.getBody());
  Value iv = loop.getInductionVar();

  DmaLoadOp::create(builder, x.getResult(), iv, c0);
  DmaLoadOp::create(builder, y.getResult(), iv, c1);
  auto xv = VPMReadOp::create(builder, vecType, c0);
  auto yv = VPMReadOp::create(builder, vecType, c1);
  auto mul = FMulOp::create(builder, vecType, a.getResult(), xv.getResult());
  auto sum = FAddOp::create(builder, vecType, mul.getResult(), yv.getResult());
  VPMWriteOp::create(builder, c1, sum.getResult());
  DmaStoreOp::create(builder, y.getResult(), iv, c1);

  builder.setInsertionPointToEnd(entry);
  gpu::ReturnOp::create(builder);
}

struct LowerGPUSaxpyToVC4Pass
    : public PassWrapper<LowerGPUSaxpyToVC4Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGPUSaxpyToVC4Pass)

  StringRef getArgument() const final { return "lower-gpu-saxpy-to-vc4"; }
  StringRef getDescription() const final {
    return "Lower a constrained logical gpu.func SAXPY kernel to a VC4-oriented persistent-worker form";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, cf::ControlFlowDialect, gpu::GPUDialect,
                    memref::MemRefDialect, scf::SCFDialect, VC4Dialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<gpu::GPUModuleOp> gpuModules;
    module.walk([&](gpu::GPUModuleOp gpuModule) { gpuModules.push_back(gpuModule); });
    if (gpuModules.size() != 1) {
      module.emitError("expected exactly one gpu.module for lower-gpu-saxpy-to-vc4");
      signalPassFailure();
      return;
    }

    SmallVector<gpu::GPUFuncOp> gpuFuncs;
    gpuModules.front().walk([&](gpu::GPUFuncOp func) { gpuFuncs.push_back(func); });
    if (gpuFuncs.size() != 1) {
      gpuModules.front().emitError(
          "expected exactly one gpu.func inside the gpu.module for lower-gpu-saxpy-to-vc4");
      signalPassFailure();
      return;
    }

    gpu::GPUFuncOp func = gpuFuncs.front();
    if (failed(validateSupportedKernel(func))) {
      signalPassFailure();
      return;
    }

    lowerKernelBody(func);
  }
};

} // namespace

std::unique_ptr<Pass> createLowerGPUSaxpyToVC4Pass() {
  return std::make_unique<LowerGPUSaxpyToVC4Pass>();
}

void registerGPUToVC4Passes() {
  PassRegistration<LowerGPUSaxpyToVC4Pass>();
}

} // namespace mlir::vc4
