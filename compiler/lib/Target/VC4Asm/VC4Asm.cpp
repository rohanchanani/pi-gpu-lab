#include "vc4/Target/VC4Asm/VC4Asm.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "vc4/Dialect/VC4/VC4Dialect.h"
#include "vc4/Dialect/VC4/VC4Ops.h"

namespace mlir::vc4 {
namespace {

class VC4AsmEmitter {
public:
  VC4AsmEmitter(llvm::raw_ostream &output) : output(output) {}

  LogicalResult emit(Operation *op) {
    output << "; vc4asm-like output generated from staged vc4 IR\n";
    output << "; contract: schematic printer only, not executable codegen\n";
    output << "; placeholders are marked explicitly where exact vc4asm syntax is TBD\n\n";
    return emitTopLevel(op);
  }

private:
  enum class SectionKind { None, Uniforms, DMALoadStore, VPM, Compute };

  LogicalResult emitTopLevel(Operation *op) {
    if (auto module = dyn_cast<ModuleOp>(op))
      return emitBlock(module.getBodyRegion().front());
    return op->emitError("mlir-to-vc4asm expects a builtin.module");
  }

  LogicalResult emitBlock(Block &block) {
    for (Operation &op : block) {
      if (isa<ModuleOp>(op))
        continue;
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        output << "; function @" << func.getName() << "\n";
        if (failed(emitBlock(func.getBody().front())))
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
    if (auto getUniform = dyn_cast<GetUniformOp>(op))
      return emitGetUniform(getUniform);
    if (auto dmaLoad = dyn_cast<DmaLoadOp>(op))
      return emitDmaLoad(dmaLoad);
    if (auto dmaStore = dyn_cast<DmaStoreOp>(op))
      return emitDmaStore(dmaStore);
    if (auto vpmRead = dyn_cast<VPMReadOp>(op))
      return emitVPMRead(vpmRead);
    if (auto vpmWrite = dyn_cast<VPMWriteOp>(op))
      return emitVPMWrite(vpmWrite);
    if (auto fmul = dyn_cast<FMulOp>(op))
      return emitFMul(fmul);
    if (auto fadd = dyn_cast<FAddOp>(op))
      return emitFAdd(fadd);
    if (auto func = dyn_cast<func::FuncOp>(op))
      return emitBlock(func.getBody().front());
    return op.emitError("unsupported op for mlir-to-vc4asm translation");
  }

  LogicalResult recordConstant(arith::ConstantOp op) {
    auto integer = dyn_cast<IntegerAttr>(op.getValue());
    if (!integer)
      return op.emitError("only integer arith.constant is supported by mlir-to-vc4asm");
    constantValues[op.getResult()] = integer.getInt();
    return success();
  }

  LogicalResult emitGetUniform(GetUniformOp op) {
    switchSection(SectionKind::Uniforms, "; Uniform reads");
    std::string symbol = getValueSymbol(op.getResult());
    output << "; " << symbol << " = vc4.get_uniform[" << op.getIndex() << "] : ";
    op.getResult().getType().print(output);
    output << "\n";
    output << "mov " << symbol << ", unif";
    output << "    ; schematic: destination register allocation is TBD\n\n";
    return success();
  }

  LogicalResult emitDmaLoad(DmaLoadOp op) {
    switchSection(SectionKind::DMALoadStore, "; DMA staging");
    output << "; vc4.dma_load " << getValueSymbol(op.getSource()) << "["
           << formatIndexValue(op.getGlobalOffset()) << "] to row "
           << formatIndexValue(op.getRow()) << "\n";
    output << "; schematic vc4asm for global-memory -> VPM staging\n";
    output << ";   mov vr_setup, vdr_setup_0(... row="
           << formatIndexValue(op.getRow()) << " ...)\n";
    output << ";   mov vr_addr, " << getValueSymbol(op.getSource()) << " + "
           << formatIndexValue(op.getGlobalOffset()) << "\n";
    output << ";   mov -, vr_wait\n\n";
    return success();
  }

  LogicalResult emitDmaStore(DmaStoreOp op) {
    switchSection(SectionKind::DMALoadStore, "; DMA staging");
    output << "; vc4.dma_store row " << formatIndexValue(op.getRow()) << " to "
           << getValueSymbol(op.getTarget()) << "["
           << formatIndexValue(op.getGlobalOffset()) << "]\n";
    output << "; schematic vc4asm for VPM -> global-memory staging\n";
    output << ";   mov vw_setup, vdw_setup_0(... row="
           << formatIndexValue(op.getRow()) << " ...)\n";
    output << ";   mov vw_addr, " << getValueSymbol(op.getTarget()) << " + "
           << formatIndexValue(op.getGlobalOffset()) << "\n";
    output << ";   mov -, vw_wait\n\n";
    return success();
  }

  LogicalResult emitVPMRead(VPMReadOp op) {
    switchSection(SectionKind::VPM, "; VPM access");
    std::string symbol = getValueSymbol(op.getResult());
    output << "; " << symbol << " = vc4.vpm_read from row "
           << formatIndexValue(op.getRow()) << "\n";
    output << "; schematic vc4asm for VPM -> QPU vector read\n";
    output << ";   mov " << symbol
           << ", vpm    ; exact destination register choice is TBD\n\n";
    return success();
  }

  LogicalResult emitVPMWrite(VPMWriteOp op) {
    switchSection(SectionKind::VPM, "; VPM access");
    output << "; vc4.vpm_write " << getValueSymbol(op.getValue()) << " to row "
           << formatIndexValue(op.getRow()) << "\n";
    output << "; schematic vc4asm for QPU vector -> VPM write\n";
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

  std::string formatIndexValue(Value value) {
    auto it = constantValues.find(value);
    if (it != constantValues.end())
      return std::to_string(it->second);
    return getValueSymbol(value);
  }

  std::string getValueSymbol(Value value) {
    auto it = valueSymbols.find(value);
    if (it != valueSymbols.end())
      return it->second;

    std::string symbol;
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      symbol = (llvm::Twine("arg") + llvm::Twine(blockArg.getArgNumber())).str();
    } else if (auto definingUniform = value.getDefiningOp<GetUniformOp>()) {
      symbol = (llvm::Twine("uniform_") + llvm::Twine(definingUniform.getIndex())).str();
    } else {
      symbol = (llvm::Twine("tmp") + llvm::Twine(nextTemporaryId++)).str();
    }

    valueSymbols[value] = symbol;
    return symbol;
  }

  llvm::raw_ostream &output;
  SectionKind currentSection = SectionKind::None;
  int nextTemporaryId = 0;
  llvm::DenseMap<Value, int64_t> constantValues;
  llvm::DenseMap<Value, std::string> valueSymbols;
};

} // namespace

void registerToVC4AsmTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-vc4asm", "Translate staged VC4 MLIR to schematic vc4asm-like text",
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
