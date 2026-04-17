#include "vc4/Target/VC4Asm/VC4Asm.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "vc4/Dialect/VC4/VC4Dialect.h"
#include "vc4/Target/KernelModel.h"

namespace mlir::vc4 {
namespace {

class VC4QASMEmitter {
public:
  VC4QASMEmitter(llvm::raw_ostream &output) : output(output) {}

  LogicalResult emit(Operation *op) {
    context = op->getContext();
    FailureOr<LoweredKernelArtifactModel> model =
        extractLoweredKernelArtifactModel(op);
    if (failed(model))
      return failure();
    return emitModel(*model);
  }

  LogicalResult emit(const LoweredKernelArtifactModel &kernelModel,
                     MLIRContext *ctx) {
    context = ctx;
    return emitModel(kernelModel);
  }

private:
  static bool isPowerOfTwo(unsigned value) { return value && ((value & (value - 1)) == 0); }

  static unsigned log2Exact(unsigned value) {
    unsigned shift = 0;
    while (value > 1) {
      value >>= 1;
      ++shift;
    }
    return shift;
  }

  static unsigned rowOffsetForDMALoad(unsigned row) { return row << 4; }
  static unsigned rowOffsetForDMAStore(unsigned row) { return row << 7; }

  std::string ra(unsigned index) const {
    return (llvm::Twine("ra") + llvm::Twine(index)).str();
  }

  std::string rb(unsigned index) const {
    return (llvm::Twine("rb") + llvm::Twine(index)).str();
  }

  std::string streamRegister(unsigned slot) const { return ra(slot); }

  std::string scalarOffsetRegister() const { return ra(model->uniformStream.size()); }
  std::string scalarStrideRegister() const { return ra(model->uniformStream.size() + 1); }
  std::string scalarExtentRegister() const { return ra(model->uniformStream.size() + 2); }

  std::string tempRegister(const KernelValueRef &value) const {
    return rb(value.index);
  }

  FailureOr<std::string> registerForValue(const KernelValueRef &value) const {
    switch (value.kind) {
    case KernelValueKind::UniformSlot:
      return streamRegister(value.index);
    case KernelValueKind::Builtin: {
      auto it = builtinRegisters.find(value.builtinKind);
      if (it == builtinRegisters.end())
        return failure();
      return it->second;
    }
    case KernelValueKind::LoopIndex:
      return scalarOffsetRegister();
    case KernelValueKind::Temporary:
      return tempRegister(value);
    }
    llvm_unreachable("unknown kernel value kind");
  }

  unsigned rowForSlot(unsigned slot) const { return slot; }

  LogicalResult emitModel(const LoweredKernelArtifactModel &kernelModel) {
    model = &kernelModel;
    if (model->execution.baseMultiplier != model->execution.laneWidth ||
        model->execution.strideMultiplier != model->execution.laneWidth)
      return emitError(UnknownLoc::get(context),
                       "qasm emission currently expects base/stride multipliers to match the lane width");

    unsigned byteStep = model->execution.laneWidth * 4;
    if (!isPowerOfTwo(byteStep))
      return emitError(UnknownLoc::get(context),
                       "qasm emission currently expects lane_width * 4 to be a power of two");

    output << ".include \"share/vc4inc/vc4.qinc\"\n\n";
    output << "# qasm emitted from lowered vc4 kernel model\n";
    output << "# kernel @" << model->kernelName << "\n\n";

    emitUniformReads();
    emitExecutionSetup(log2Exact(byteStep));
    emitInitialGuard();
    output << ":loop\n";
    if (failed(emitBody()))
      return failure();
    emitLoopAdvance();
    emitThreadEnd();
    return success();
  }

  void emitUniformReads() {
    for (const UniformStreamEntry &entry : model->uniformStream) {
      std::string reg = streamRegister(entry.slot);
      if (entry.sourceKind == UniformSourceKind::PublicArgument) {
        output << "# uniform[" << entry.slot << "] = arg" << entry.publicArgumentIndex
               << "\n";
      } else {
        output << "# uniform[" << entry.slot << "] = builtin "
               << stringifyBuiltinKind(entry.builtinKind) << "\n";
        builtinRegisters[entry.builtinKind] = reg;
      }
      output << "mov " << reg << ", unif\n";
    }
    output << "\n";
  }

  void emitExecutionSetup(unsigned byteShift) {
    output << "# base = qpu_id * " << model->execution.laneWidth
           << " elements; stride = num_qpus * " << model->execution.laneWidth
           << " elements\n";
    output << "# qasm uses byte-addressed DMA pointers, so the loop offset is scaled by 4\n";
    output << "shl r0, " << builtinRegisters.lookup(model->execution.workerIdBuiltin)
           << ", " << byteShift << "\n";
    output << "mov " << scalarOffsetRegister() << ", r0\n";
    output << "shl r0, " << builtinRegisters.lookup(model->execution.workerCountBuiltin)
           << ", " << byteShift << "\n";
    output << "mov " << scalarStrideRegister() << ", r0\n";
    output << "shl r0, " << streamRegister(model->execution.upperBoundUniformSlot)
           << ", 2\n";
    output << "mov " << scalarExtentRegister() << ", r0\n\n";
  }

  void emitInitialGuard() {
    output << "# Skip execution if this worker's initial chunk is already out of range\n";
    output << "mov r1, " << scalarExtentRegister() << "\n";
    output << "sub.setf r1, " << scalarOffsetRegister() << ", r1\n";
    output << "brr.anync -, :end\n";
    output << "nop\n";
    output << "nop\n";
    output << "nop\n\n";
  }

  LogicalResult emitBody() {
    for (const BodyOperationInfo &op : model->body) {
      switch (op.kind) {
      case BodyOperationKind::DmaLoad:
        emitDmaLoad(op);
        break;
      case BodyOperationKind::StagedRead:
        if (failed(emitStagedRead(op)))
          return failure();
        break;
      case BodyOperationKind::FMul:
        if (failed(emitFMul(op)))
          return failure();
        break;
      case BodyOperationKind::FAdd:
        if (failed(emitFAdd(op)))
          return failure();
        break;
      case BodyOperationKind::StagedWrite:
        if (failed(emitStagedWrite(op)))
          return failure();
        break;
      case BodyOperationKind::DmaStore:
        emitDmaStore(op);
        break;
      }
    }
    return success();
  }

  void emitDmaLoad(const BodyOperationInfo &op) {
    unsigned row = rowForSlot(*op.stagingSlot);
    output << "    # dma_load uniform" << *op.memrefUniformSlot << "[iv] -> stage"
           << *op.stagingSlot << " (emitter-local row " << row << ")\n";
    output << "    mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0))\n";
    if (row == 0)
      output << "    mov vr_setup, r2\n";
    else
      output << "    add vr_setup, r2, " << rowOffsetForDMALoad(row) << "\n";
    output << "    add vr_addr, " << streamRegister(*op.memrefUniformSlot) << ", "
           << scalarOffsetRegister() << "\n";
    output << "    mov -, vr_wait\n\n";
  }

  LogicalResult emitStagedRead(const BodyOperationInfo &op) {
    unsigned row = rowForSlot(*op.stagingSlot);
    FailureOr<std::string> result = registerForValue(*op.result);
    if (failed(result))
      return emitError(UnknownLoc::get(context),
                       "qasm emission failed to assign a register to staged_read result");
    output << "    # staged_read stage" << *op.stagingSlot << " -> "
           << *result << " (emitter-local row " << row << ")\n";
    output << "    mov r2, vpm_setup(1, 1, h32(0))\n";
    if (row == 0)
      output << "    mov vr_setup, r2\n";
    else
      output << "    add vr_setup, r2, " << row << "\n";
    output << "    mov " << *result << ", vpm\n";
    output << "    mov -, vw_wait\n\n";
    return success();
  }

  LogicalResult emitFMul(const BodyOperationInfo &op) {
    FailureOr<std::string> result = registerForValue(*op.result);
    FailureOr<std::string> lhs = registerForValue(*op.lhs);
    FailureOr<std::string> rhs = registerForValue(*op.rhs);
    if (failed(result) || failed(lhs) || failed(rhs))
      return emitError(UnknownLoc::get(context),
                       "qasm emission failed to assign registers for fmul");
    output << "    fmul " << *result << ", " << *lhs << ", " << *rhs << "\n";
    return success();
  }

  LogicalResult emitFAdd(const BodyOperationInfo &op) {
    FailureOr<std::string> result = registerForValue(*op.result);
    FailureOr<std::string> lhs = registerForValue(*op.lhs);
    FailureOr<std::string> rhs = registerForValue(*op.rhs);
    if (failed(result) || failed(lhs) || failed(rhs))
      return emitError(UnknownLoc::get(context),
                       "qasm emission failed to assign registers for fadd");
    output << "    fadd " << *result << ", " << *lhs << ", " << *rhs << "\n";
    return success();
  }

  LogicalResult emitStagedWrite(const BodyOperationInfo &op) {
    unsigned row = rowForSlot(*op.stagingSlot);
    FailureOr<std::string> source = registerForValue(*op.source);
    if (failed(source))
      return emitError(UnknownLoc::get(context),
                       "qasm emission failed to assign a register for staged_write");
    output << "    # staged_write " << *source << " -> stage" << *op.stagingSlot
           << " (emitter-local row " << row << ")\n";
    output << "    mov r2, vpm_setup(1, 1, h32(0))\n";
    if (row == 0)
      output << "    mov vw_setup, r2\n";
    else
      output << "    add vw_setup, r2, " << row << "\n";
    output << "    mov vpm, " << *source << "\n";
    output << "    mov -, vw_wait\n\n";
    return success();
  }

  void emitDmaStore(const BodyOperationInfo &op) {
    unsigned row = rowForSlot(*op.stagingSlot);
    output << "    # dma_store stage" << *op.stagingSlot << " -> uniform"
           << *op.memrefUniformSlot << "[iv] (emitter-local row " << row << ")\n";
    output << "    mov r2, vdw_setup_0(1, 16, dma_h32(0, 0))\n";
    if (row == 0)
      output << "    mov vw_setup, r2\n";
    else
      output << "    add vw_setup, r2, " << rowOffsetForDMAStore(row) << "\n";
    output << "    add vw_addr, " << streamRegister(*op.memrefUniformSlot) << ", "
           << scalarOffsetRegister() << "\n";
    output << "    mov -, vw_wait\n\n";
  }

  void emitLoopAdvance() {
    output << "    add " << scalarOffsetRegister() << ", " << scalarOffsetRegister()
           << ", " << scalarStrideRegister() << "\n";
    output << "    mov r1, " << scalarExtentRegister() << "\n";
    output << "    sub.setf r1, " << scalarOffsetRegister() << ", r1\n";
    output << "    brr.anyc -, :loop\n";
    output << "    nop\n";
    output << "    nop\n";
    output << "    nop\n\n";
  }

  void emitThreadEnd() {
    output << ":end\n";
    output << "thrend\n";
    output << "mov interrupt, 1\n";
    output << "nop\n";
    output << "nop\n";
  }

  llvm::raw_ostream &output;
  MLIRContext *context = nullptr;
  const LoweredKernelArtifactModel *model = nullptr;
  llvm::DenseMap<BuiltinKind, std::string> builtinRegisters;
};

} // namespace

LogicalResult emitVC4QASM(const LoweredKernelArtifactModel &model,
                          llvm::raw_ostream &output) {
  VC4QASMEmitter emitter(output);
  return emitter.emit(model, nullptr);
}

LogicalResult emitVC4QASM(Operation *op, llvm::raw_ostream &output) {
  VC4QASMEmitter emitter(output);
  return emitter.emit(op);
}

void registerToVC4QASMTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-vc4-qasm",
      "Translate lowered VC4 MLIR to concrete vc4asm/qasm text for the current supported subset",
      [](Operation *op, llvm::raw_ostream &output) {
        return emitVC4QASM(op, output);
      },
      [](DialectRegistry &registry) {
        registerAllDialects(registry);
        registry.insert<VC4Dialect>();
      });
}

} // namespace mlir::vc4
