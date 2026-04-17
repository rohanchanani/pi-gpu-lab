#include "vc4/Target/VC4Asm/VC4Asm.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "vc4/Dialect/VC4/VC4Dialect.h"
#include "vc4/Dialect/VC4/VC4Ops.h"
#include "vc4/Target/KernelModel.h"

#include <optional>

namespace mlir::vc4 {
namespace {

class VC4AsmEmitter {
public:
  VC4AsmEmitter(llvm::raw_ostream &output) : output(output) {}

  LogicalResult emit(Operation *op) {
    output << "; inspection-only VC4 assembly sketch generated from lowered vc4 IR\n";
    output << "; contract: intermediate inspection output, not the final artifact boundary\n";
    output << "; placeholders are marked explicitly where exact VC4 assembly syntax is still TBD\n\n";

    if (std::optional<gpu::GPUFuncOp> kernelFunc = findSingleKernelFunc(op))
      return emitLoweredKernel(*kernelFunc);

    return emitTopLevel(op);
  }

private:
  enum class SectionKind {
    None,
    Uniforms,
    BuiltinSuffix,
    ScalarSetup,
    ControlFlow,
    DMALoadStore,
    Staging,
    Compute
  };

  std::optional<gpu::GPUFuncOp> findSingleKernelFunc(Operation *op) {
    if (auto func = dyn_cast<gpu::GPUFuncOp>(op)) {
      if (func->hasAttr("gpu.kernel"))
        return func;
      return std::nullopt;
    }

    llvm::SmallVector<gpu::GPUFuncOp> funcs;
    op->walk([&](gpu::GPUFuncOp func) {
      if (func->hasAttr("gpu.kernel"))
        funcs.push_back(func);
    });
    if (funcs.size() == 1)
      return funcs.front();
    return std::nullopt;
  }

  LogicalResult emitLoweredKernel(gpu::GPUFuncOp func) {
    FailureOr<LoweredKernelArtifactModel> model =
        extractLoweredKernelArtifactModel(func);
    if (failed(model))
      return failure();

    if (auto gpuModule = func->getParentOfType<gpu::GPUModuleOp>())
      output << "; gpu.module @" << gpuModule.getName() << "\n";
    output << "; gpu.func @" << func.getName() << "\n";

    if (failed(emitLoweredKernelUniformStream(*model)))
      return failure();
    emitLoweredKernelExecution(*model);
    if (failed(emitLoweredKernelBody(*model)))
      return failure();

    output << "\n";
    return success();
  }

  LogicalResult emitLoweredKernelUniformStream(
      const LoweredKernelArtifactModel &model) {
    for (const UniformStreamEntry &entry : model.uniformStream) {
      if (entry.sourceKind == UniformSourceKind::PublicArgument) {
        switchSection(SectionKind::Uniforms, "; Uniform reads");
        std::string symbol = uniformSlotSymbol(entry.slot);
        output << "; " << symbol << " = vc4.get_uniform[" << entry.slot << "] : ";
        entry.type.print(output);
        output << "\n";
        output << "mov " << symbol
               << ", unif    ; physical stream word " << entry.slot
               << " in source argument order\n\n";
        continue;
      }

      switchSection(SectionKind::BuiltinSuffix, "; Builtin suffix reads");
      std::string symbol = builtinSymbol(entry.builtinKind);
      output << "; " << symbol << " = vc4.get_builtin "
             << stringifyBuiltinKind(entry.builtinKind) << " : ";
      entry.type.print(output);
      output << "\n";
      output << "mov " << symbol
             << ", unif    ; builtin delivered through reserved uniform suffix word "
             << entry.slot << "\n\n";
    }

    return success();
  }

  void emitLoweredKernelExecution(const LoweredKernelArtifactModel &model) {
    switchSection(SectionKind::ScalarSetup, "; Scalar/index setup");
    output << "; base = " << builtinSymbol(model.execution.workerIdBuiltin) << " * "
           << model.execution.baseMultiplier << "\n";
    output << "; stride = " << builtinSymbol(model.execution.workerCountBuiltin) << " * "
           << model.execution.strideMultiplier << "\n";
    output << "; lane_width = " << model.execution.laneWidth << "\n\n";

    switchSection(SectionKind::ControlFlow, "; Control");
    output << "; scf.for iv0 = base to "
           << uniformSlotSymbol(model.execution.upperBoundUniformSlot)
           << " step stride\n";
    if (model.execution.assumesUpperBoundMultipleOfLaneWidth) {
      output << "; current lowered subset assumes the upper bound is a multiple of "
             << model.execution.laneWidth << " (no explicit tail guard emitted)\n";
    }
    output << "\n";
  }

  LogicalResult emitLoweredKernelBody(const LoweredKernelArtifactModel &model) {
    for (const BodyOperationInfo &op : model.body) {
      switch (op.kind) {
      case BodyOperationKind::DmaLoad:
        emitModelDmaLoad(op);
        break;
      case BodyOperationKind::StagedRead:
        emitModelStagedRead(op);
        break;
      case BodyOperationKind::FMul:
        emitModelFMul(op);
        break;
      case BodyOperationKind::FAdd:
        emitModelFAdd(op);
        break;
      case BodyOperationKind::StagedWrite:
        emitModelStagedWrite(op);
        break;
      case BodyOperationKind::DmaStore:
        emitModelDmaStore(op);
        break;
      }
    }
    return success();
  }

  void emitModelDmaLoad(const BodyOperationInfo &op) {
    unsigned slot = *op.stagingSlot;
    unsigned row = physicalRowForStageSlot(slot);
    switchSection(SectionKind::DMALoadStore, "; DMA staging");
    output << "; vc4.dma_load " << uniformSlotSymbol(*op.memrefUniformSlot) << "["
           << formatKernelValueRef(*op.offset) << "] to slot[" << slot << "]\n";
    output << "; debug-only emitter-local mapping: slot[" << slot
           << "] currently prints as physical row " << row << "\n";
    output << "; inspection sketch for global-memory -> VPM staging\n";
    output << ";   mov vr_setup, vdr_setup_0(... row=" << row << " ...)\n";
    output << ";   mov vr_addr, " << uniformSlotSymbol(*op.memrefUniformSlot) << " + "
           << formatKernelValueRef(*op.offset) << "\n";
    output << ";   mov -, vr_wait\n\n";
  }

  void emitModelDmaStore(const BodyOperationInfo &op) {
    unsigned slot = *op.stagingSlot;
    unsigned row = physicalRowForStageSlot(slot);
    switchSection(SectionKind::DMALoadStore, "; DMA staging");
    output << "; vc4.dma_store slot[" << slot << "] to "
           << uniformSlotSymbol(*op.memrefUniformSlot) << "["
           << formatKernelValueRef(*op.offset) << "]\n";
    output << "; debug-only emitter-local mapping: slot[" << slot
           << "] currently prints as physical row " << row << "\n";
    output << "; inspection sketch for VPM -> global-memory staging\n";
    output << ";   mov vw_setup, vdw_setup_0(... row=" << row << " ...)\n";
    output << ";   mov vw_addr, " << uniformSlotSymbol(*op.memrefUniformSlot) << " + "
           << formatKernelValueRef(*op.offset) << "\n";
    output << ";   mov -, vw_wait\n\n";
  }

  void emitModelStagedRead(const BodyOperationInfo &op) {
    unsigned slot = *op.stagingSlot;
    unsigned row = physicalRowForStageSlot(slot);
    switchSection(SectionKind::Staging, "; Staging access");
    output << "; " << formatKernelValueRef(*op.result)
           << " = vc4.staged_read from slot[" << slot << "]\n";
    output << "; debug-only emitter-local mapping: slot[" << slot
           << "] currently prints as physical row " << row << "\n";
    output << "; inspection sketch for VPM -> QPU vector read\n";
    output << ";   mov " << formatKernelValueRef(*op.result)
           << ", vpm    ; exact destination register choice is TBD\n\n";
  }

  void emitModelStagedWrite(const BodyOperationInfo &op) {
    unsigned slot = *op.stagingSlot;
    unsigned row = physicalRowForStageSlot(slot);
    switchSection(SectionKind::Staging, "; Staging access");
    output << "; vc4.staged_write slot[" << slot << "] = "
           << formatKernelValueRef(*op.source) << "\n";
    output << "; debug-only emitter-local mapping: slot[" << slot
           << "] currently prints as physical row " << row << "\n";
    output << "; inspection sketch for QPU vector -> VPM write\n";
    output << ";   mov vpm, " << formatKernelValueRef(*op.source)
           << "    ; exact source register choice is TBD\n\n";
  }

  void emitModelFMul(const BodyOperationInfo &op) {
    switchSection(SectionKind::Compute, "; Compute");
    output << "; " << formatKernelValueRef(*op.result) << " = vc4.fmul "
           << formatKernelValueRef(*op.lhs) << ", "
           << formatKernelValueRef(*op.rhs) << "\n";
    output << "fmul " << formatKernelValueRef(*op.result) << ", "
           << formatKernelValueRef(*op.lhs) << ", "
           << formatKernelValueRef(*op.rhs)
           << "    ; schematic: accumulator / register-file choice is TBD\n\n";
  }

  void emitModelFAdd(const BodyOperationInfo &op) {
    switchSection(SectionKind::Compute, "; Compute");
    output << "; " << formatKernelValueRef(*op.result) << " = vc4.fadd "
           << formatKernelValueRef(*op.lhs) << ", "
           << formatKernelValueRef(*op.rhs) << "\n";
    output << "fadd " << formatKernelValueRef(*op.result) << ", "
           << formatKernelValueRef(*op.lhs) << ", "
           << formatKernelValueRef(*op.rhs)
           << "    ; schematic: accumulator / register-file choice is TBD\n\n";
  }

  LogicalResult emitTopLevel(Operation *op) {
    if (auto module = dyn_cast<ModuleOp>(op))
      return emitBlock(module.getBodyRegion().front());
    return op->emitError("mlir-to-vc4asm expects a builtin.module or lowered gpu.func");
  }

  LogicalResult emitBlock(Block &block) {
    for (Operation &op : block) {
      if (isa<ModuleOp>(op))
        continue;
      if (auto gpuModule = dyn_cast<gpu::GPUModuleOp>(op)) {
        output << "; gpu.module @" << gpuModule.getName() << "\n";
        if (failed(emitBlock(gpuModule.getBodyRegion().front())))
          return failure();
        output << "\n";
        continue;
      }
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        output << "; function @" << func.getName() << "\n";
        if (failed(emitBlock(func.getBody().front())))
          return failure();
        output << "\n";
        continue;
      }
      if (auto gpuFunc = dyn_cast<gpu::GPUFuncOp>(op)) {
        output << "; gpu.func @" << gpuFunc.getName() << "\n";
        if (failed(emitBlock(gpuFunc.getBody().front())))
          return failure();
        output << "\n";
        continue;
      }
      if (failed(emitOp(op)))
        return failure();
    }
    return success();
  }

  LogicalResult emitOp(Operation &op) {
    if (auto constant = dyn_cast<arith::ConstantOp>(op))
      return recordConstant(constant);
    if (auto muli = dyn_cast<arith::MulIOp>(op))
      return emitMulI(muli);
    if (auto remui = dyn_cast<arith::RemUIOp>(op))
      return emitRemUI(remui);
    if (auto cmpi = dyn_cast<arith::CmpIOp>(op))
      return emitCmpI(cmpi);
    if (auto assertOp = dyn_cast<cf::AssertOp>(op))
      return emitAssert(assertOp);
    if (auto forOp = dyn_cast<scf::ForOp>(op))
      return emitFor(forOp);
    if (auto getBuiltin = dyn_cast<GetBuiltinOp>(op))
      return emitGetBuiltin(getBuiltin);
    if (auto getUniform = dyn_cast<GetUniformOp>(op))
      return emitGetUniform(getUniform);
    if (auto dmaLoad = dyn_cast<DmaLoadOp>(op))
      return emitDmaLoad(dmaLoad);
    if (auto dmaStore = dyn_cast<DmaStoreOp>(op))
      return emitDmaStore(dmaStore);
    if (auto stagedRead = dyn_cast<StagedReadOp>(op))
      return emitStagedRead(stagedRead);
    if (auto stagedWrite = dyn_cast<StagedWriteOp>(op))
      return emitStagedWrite(stagedWrite);
    if (auto fmul = dyn_cast<FMulOp>(op))
      return emitFMul(fmul);
    if (auto fadd = dyn_cast<FAddOp>(op))
      return emitFAdd(fadd);
    if (auto func = dyn_cast<func::FuncOp>(op))
      return emitBlock(func.getBody().front());
    if (auto gpuFunc = dyn_cast<gpu::GPUFuncOp>(op))
      return emitBlock(gpuFunc.getBody().front());
    if (isa<gpu::ReturnOp>(op) || isa<scf::YieldOp>(op))
      return success();
    return op.emitError("unsupported op for mlir-to-vc4asm translation");
  }

  LogicalResult recordConstant(arith::ConstantOp op) {
    auto integer = dyn_cast<IntegerAttr>(op.getValue());
    if (!integer)
      return op.emitError("only integer arith.constant is supported by mlir-to-vc4asm");
    constantValues[op.getResult()] = integer.getInt();
    return success();
  }

  LogicalResult emitMulI(arith::MulIOp op) {
    switchSection(SectionKind::ScalarSetup, "; Scalar/index setup");
    output << "; " << getValueSymbol(op.getResult()) << " = arith.muli "
           << formatIndexValue(op.getLhs()) << ", " << formatIndexValue(op.getRhs())
           << "\n";
    return success();
  }

  LogicalResult emitRemUI(arith::RemUIOp op) {
    switchSection(SectionKind::ScalarSetup, "; Scalar/index setup");
    output << "; " << getValueSymbol(op.getResult()) << " = arith.remui "
           << formatIndexValue(op.getLhs()) << ", " << formatIndexValue(op.getRhs())
           << "\n";
    return success();
  }

  LogicalResult emitCmpI(arith::CmpIOp op) {
    switchSection(SectionKind::ScalarSetup, "; Scalar/index setup");
    output << "; " << getValueSymbol(op.getResult()) << " = arith.cmpi "
           << stringifyCmpIPredicate(op.getPredicate()) << ", "
           << formatIndexValue(op.getLhs()) << ", " << formatIndexValue(op.getRhs())
           << "\n";
    return success();
  }

  LogicalResult emitAssert(cf::AssertOp op) {
    switchSection(SectionKind::ControlFlow, "; Control");
    output << "; cf.assert " << getValueSymbol(op.getArg()) << ", \""
           << op.getMsg() << "\"\n";
    return success();
  }

  LogicalResult emitFor(scf::ForOp op) {
    switchSection(SectionKind::ControlFlow, "; Control");
    std::string ivName = (llvm::Twine("iv") + llvm::Twine(nextLoopId++)).str();
    valueSymbols[op.getInductionVar()] = ivName;
    output << "; scf.for " << ivName << " = " << formatIndexValue(op.getLowerBound())
           << " to " << formatIndexValue(op.getUpperBound()) << " step "
           << formatIndexValue(op.getStep()) << "\n";
    if (failed(emitBlock(op.getRegion().front())))
      return failure();
    output << "; end scf.for " << ivName << "\n\n";
    return success();
  }

  LogicalResult emitGetBuiltin(GetBuiltinOp op) {
    switchSection(SectionKind::BuiltinSuffix, "; Builtin suffix reads");
    std::string symbol = getValueSymbol(op.getResult());
    output << "; " << symbol << " = vc4.get_builtin "
           << stringifyBuiltinKind(op.getBuiltinKind()) << " : index\n";
    output << "mov " << symbol
           << ", unif    ; schematic: builtin value arrives via reserved uniform suffix\n\n";
    return success();
  }

  LogicalResult emitGetUniform(GetUniformOp op) {
    switchSection(SectionKind::Uniforms, "; Uniform reads");
    std::string symbol = getValueSymbol(op.getResult());
    output << "; " << symbol << " = vc4.get_uniform[" << op.getIndex() << "] : ";
    op.getResult().getType().print(output);
    output << "\n";
    output << "mov " << symbol
           << ", unif    ; schematic: destination register allocation is TBD\n\n";
    return success();
  }

  LogicalResult emitDmaLoad(DmaLoadOp op) {
    unsigned slot = op.getSlot();
    unsigned row = physicalRowForStageSlot(slot);
    switchSection(SectionKind::DMALoadStore, "; DMA staging");
    output << "; vc4.dma_load " << getValueSymbol(op.getSource()) << "["
           << formatIndexValue(op.getGlobalOffset()) << "] to slot[" << slot << "]\n";
    output << "; debug-only emitter-local mapping: slot[" << slot
           << "] currently prints as physical row " << row << "\n";
    output << "; inspection sketch for global-memory -> VPM staging\n";
    output << ";   mov vr_setup, vdr_setup_0(... row=" << row << " ...)\n";
    output << ";   mov vr_addr, " << getValueSymbol(op.getSource()) << " + "
           << formatIndexValue(op.getGlobalOffset()) << "\n";
    output << ";   mov -, vr_wait\n\n";
    return success();
  }

  LogicalResult emitDmaStore(DmaStoreOp op) {
    unsigned slot = op.getSlot();
    unsigned row = physicalRowForStageSlot(slot);
    switchSection(SectionKind::DMALoadStore, "; DMA staging");
    output << "; vc4.dma_store slot[" << slot << "] to "
           << getValueSymbol(op.getTarget()) << "["
           << formatIndexValue(op.getGlobalOffset()) << "]\n";
    output << "; debug-only emitter-local mapping: slot[" << slot
           << "] currently prints as physical row " << row << "\n";
    output << "; inspection sketch for VPM -> global-memory staging\n";
    output << ";   mov vw_setup, vdw_setup_0(... row=" << row << " ...)\n";
    output << ";   mov vw_addr, " << getValueSymbol(op.getTarget()) << " + "
           << formatIndexValue(op.getGlobalOffset()) << "\n";
    output << ";   mov -, vw_wait\n\n";
    return success();
  }

  LogicalResult emitStagedRead(StagedReadOp op) {
    unsigned slot = op.getSlot();
    unsigned row = physicalRowForStageSlot(slot);
    switchSection(SectionKind::Staging, "; Staging access");
    std::string symbol = getValueSymbol(op.getResult());
    output << "; " << symbol << " = vc4.staged_read from slot[" << slot << "]\n";
    output << "; debug-only emitter-local mapping: slot[" << slot
           << "] currently prints as physical row " << row << "\n";
    output << "; inspection sketch for VPM -> QPU vector read\n";
    output << ";   mov " << symbol
           << ", vpm    ; exact destination register choice is TBD\n\n";
    return success();
  }

  LogicalResult emitStagedWrite(StagedWriteOp op) {
    unsigned slot = op.getSlot();
    unsigned row = physicalRowForStageSlot(slot);
    switchSection(SectionKind::Staging, "; Staging access");
    output << "; vc4.staged_write slot[" << slot << "] = "
           << getValueSymbol(op.getValue()) << "\n";
    output << "; debug-only emitter-local mapping: slot[" << slot
           << "] currently prints as physical row " << row << "\n";
    output << "; inspection sketch for QPU vector -> VPM write\n";
    output << ";   mov vpm, " << getValueSymbol(op.getValue())
           << "    ; exact source register choice is TBD\n\n";
    return success();
  }

  LogicalResult emitFMul(FMulOp op) {
    switchSection(SectionKind::Compute, "; Compute");
    std::string symbol = getValueSymbol(op.getResult());
    output << "; " << symbol << " = vc4.fmul " << getValueSymbol(op.getLhs())
           << ", " << getValueSymbol(op.getRhs()) << "\n";
    output << "fmul " << symbol << ", " << getValueSymbol(op.getLhs()) << ", "
           << getValueSymbol(op.getRhs())
           << "    ; schematic: accumulator / register-file choice is TBD\n\n";
    return success();
  }

  LogicalResult emitFAdd(FAddOp op) {
    switchSection(SectionKind::Compute, "; Compute");
    std::string symbol = getValueSymbol(op.getResult());
    output << "; " << symbol << " = vc4.fadd " << getValueSymbol(op.getLhs())
           << ", " << getValueSymbol(op.getRhs()) << "\n";
    output << "fadd " << symbol << ", " << getValueSymbol(op.getLhs()) << ", "
           << getValueSymbol(op.getRhs())
           << "    ; schematic: accumulator / register-file choice is TBD\n\n";
    return success();
  }

  void switchSection(SectionKind newSection, StringRef title) {
    if (currentSection == newSection)
      return;
    currentSection = newSection;
    output << title << "\n";
  }

  unsigned physicalRowForStageSlot(unsigned stageSlot) {
    auto it = stageSlotToPhysicalRow.find(stageSlot);
    if (it != stageSlotToPhysicalRow.end())
      return it->second;
    stageSlotToPhysicalRow[stageSlot] = stageSlot;
    return stageSlot;
  }

  std::string formatIndexValue(Value value) {
    auto it = constantValues.find(value);
    if (it != constantValues.end())
      return std::to_string(it->second);
    return getValueSymbol(value);
  }

  std::string uniformSlotSymbol(unsigned slot) const {
    return (llvm::Twine("uniform_") + llvm::Twine(slot)).str();
  }

  std::string builtinSymbol(BuiltinKind builtinKind) const {
    return (llvm::Twine("builtin_") +
            llvm::Twine(stringifyBuiltinKind(builtinKind))).str();
  }

  std::string formatKernelValueRef(const KernelValueRef &value) const {
    switch (value.kind) {
    case KernelValueKind::UniformSlot:
      return uniformSlotSymbol(value.index);
    case KernelValueKind::Builtin:
      return builtinSymbol(value.builtinKind);
    case KernelValueKind::LoopIndex:
      return "iv0";
    case KernelValueKind::Temporary:
      return (llvm::Twine("tmp") + llvm::Twine(value.index)).str();
    }
    llvm_unreachable("unknown kernel value kind");
  }

  std::string getValueSymbol(Value value) {
    auto it = valueSymbols.find(value);
    if (it != valueSymbols.end())
      return it->second;

    std::string symbol;
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      symbol = (llvm::Twine("arg") + llvm::Twine(blockArg.getArgNumber())).str();
    } else if (auto definingBuiltin = value.getDefiningOp<GetBuiltinOp>()) {
      symbol = builtinSymbol(definingBuiltin.getBuiltinKind());
    } else if (auto definingUniform = value.getDefiningOp<GetUniformOp>()) {
      symbol = uniformSlotSymbol(definingUniform.getIndex());
    } else {
      symbol = (llvm::Twine("tmp") + llvm::Twine(nextTemporaryId++)).str();
    }

    valueSymbols[value] = symbol;
    return symbol;
  }

  llvm::raw_ostream &output;
  SectionKind currentSection = SectionKind::None;
  int nextTemporaryId = 0;
  int nextLoopId = 0;
  llvm::DenseMap<Value, int64_t> constantValues;
  llvm::DenseMap<Value, std::string> valueSymbols;
  llvm::DenseMap<unsigned, unsigned> stageSlotToPhysicalRow;
};

} // namespace

void registerToVC4AsmTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-vc4asm",
      "Translate lowered VC4 MLIR to intermediate inspection-oriented VC4 assembly text",
      [](Operation *op, llvm::raw_ostream &output) {
        VC4AsmEmitter emitter(output);
        return emitter.emit(op);
      },
      [](DialectRegistry &registry) {
        registerAllDialects(registry);
        registry.insert<VC4Dialect>();
      });
}

} // namespace mlir::vc4
