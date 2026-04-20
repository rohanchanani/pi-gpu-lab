//===- vc4-opt.cpp - VC4 standalone optimizer driver ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4/IR/VC4Ops.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"

namespace {

static bool isAllowedVC4QASMInputOp(mlir::Operation &op) {
  return llvm::isa<mlir::vc4::QPUBundleOp, mlir::vc4::QPUBranchOp,
                   mlir::vc4::QPULDIOp, mlir::vc4::QPUSemaOp>(op);
}

static bool isAllowedVC4LauncherInputOp(mlir::Operation &op) {
  return llvm::isa<mlir::vc4::AsyncWaitOp, mlir::vc4::CFBranchOp,
                   mlir::vc4::EnqueueQPUOp, mlir::vc4::ReserveQPUOp,
                   mlir::vc4::V3DQueryOp, mlir::vc4::V3DConfigureOp,
                   mlir::vc4::ReturnOp>(op);
}

static bool isVC4ScheduledSinkFamilyOp(mlir::Operation &op) {
  return op.getName().getStringRef().starts_with("vc4.qpu.");
}

static bool isVC4LauncherOrSystemOp(mlir::Operation &op) {
  return llvm::isa<mlir::vc4::AsyncWaitOp, mlir::vc4::CFBranchOp,
                   mlir::vc4::EnqueueQPUOp, mlir::vc4::ReserveQPUOp,
                   mlir::vc4::V3DQueryOp, mlir::vc4::V3DConfigureOp,
                   mlir::vc4::ReturnOp>(op);
}

static bool isVC4StructuredFamilyOp(mlir::Operation &op) {
  llvm::StringRef name = op.getName().getStringRef();
  return name.starts_with("vc4.") && !name.starts_with("vc4.qpu.");
}

static bool isVC4ScheduledQPUDomainFunction(mlir::vc4::FuncOp func) {
  std::optional<mlir::vc4::ExecutionDomain> domain = func.getDomain();
  std::optional<mlir::vc4::FunctionForm> form = func.getForm();
  return domain && *domain == mlir::vc4::ExecutionDomain::qpu && form &&
         *form == mlir::vc4::FunctionForm::scheduled;
}

static bool isVC4PhysicalRegfileWriteAddress(int64_t value) {
  return value >= 0 && value <= 31;
}

static constexpr int64_t kVC4QPUWriteAddressR5 = 37;
static constexpr int64_t kVC4QPUSFUWriteAddressMin = 52;
static constexpr int64_t kVC4QPUSFUWriteAddressMax = 55;

static bool isVC4RegisterAddress14(int64_t value) { return value == 14; }

// In the current scheduled sink subset, uniform reads are represented by
// raddr_* = 14. VPM/VDR/VDW accesses are represented by peripheral write
// addresses 48..51.
static bool isVC4VPMOrDMAWriteAddress(int64_t value) {
  return value >= 48 && value <= 51;
}

static bool isVC4BundleAddPipeWriteActive(mlir::vc4::QPUBundleOp op) {
  return op.getOpAdd() != mlir::vc4::AddOpcode::nop;
}

static bool isVC4BundleMulPipeWriteActive(mlir::vc4::QPUBundleOp op) {
  return op.getOpMul() != mlir::vc4::MulOpcode::nop;
}

static bool isVC4ThreadEndSignal(mlir::vc4::QPUSignal signal) {
  return signal == mlir::vc4::QPUSignal::thrend;
}

static bool isVC4TMULoadSignal(mlir::vc4::QPUSignal signal) {
  return signal == mlir::vc4::QPUSignal::ldtmu0 ||
         signal == mlir::vc4::QPUSignal::ldtmu1;
}

static bool isVC4SFUWriteAddress(int64_t value) {
  return value >= kVC4QPUSFUWriteAddressMin &&
         value <= kVC4QPUSFUWriteAddressMax;
}

static bool isVC4AccumulatorR5WriteAddress(int64_t value) {
  return value == kVC4QPUWriteAddressR5;
}

static bool scheduledInstructionWritesPhysicalRegfile(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    return (isVC4BundleAddPipeWriteActive(bundle) &&
            isVC4PhysicalRegfileWriteAddress(bundle.getWaddrAddAttr().getInt())) ||
           (isVC4BundleMulPipeWriteActive(bundle) &&
            isVC4PhysicalRegfileWriteAddress(bundle.getWaddrMulAttr().getInt()));
  }
  if (auto ldi = llvm::dyn_cast<mlir::vc4::QPULDIOp>(op)) {
    return isVC4PhysicalRegfileWriteAddress(ldi.getWaddrAddAttr().getInt()) ||
           isVC4PhysicalRegfileWriteAddress(ldi.getWaddrMulAttr().getInt());
  }
  if (auto sema = llvm::dyn_cast<mlir::vc4::QPUSemaOp>(op)) {
    return isVC4PhysicalRegfileWriteAddress(sema.getWaddrAddAttr().getInt()) ||
           isVC4PhysicalRegfileWriteAddress(sema.getWaddrMulAttr().getInt());
  }
  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    return isVC4PhysicalRegfileWriteAddress(branch.getWaddrAddAttr().getInt()) ||
           isVC4PhysicalRegfileWriteAddress(branch.getWaddrMulAttr().getInt());
  }
  return false;
}

static std::optional<int64_t>
scheduledInstructionPhysicalRegfileAWrite(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    int64_t value = bundle.getWaddrAddAttr().getInt();
    if (isVC4BundleAddPipeWriteActive(bundle) &&
        isVC4PhysicalRegfileWriteAddress(value)) {
      return value;
    }
    return std::nullopt;
  }
  if (auto ldi = llvm::dyn_cast<mlir::vc4::QPULDIOp>(op)) {
    int64_t value = ldi.getWaddrAddAttr().getInt();
    if (isVC4PhysicalRegfileWriteAddress(value))
      return value;
    return std::nullopt;
  }
  if (auto sema = llvm::dyn_cast<mlir::vc4::QPUSemaOp>(op)) {
    int64_t value = sema.getWaddrAddAttr().getInt();
    if (isVC4PhysicalRegfileWriteAddress(value))
      return value;
    return std::nullopt;
  }
  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    int64_t value = branch.getWaddrAddAttr().getInt();
    if (isVC4PhysicalRegfileWriteAddress(value))
      return value;
    return std::nullopt;
  }
  return std::nullopt;
}

static std::optional<int64_t>
scheduledInstructionPhysicalRegfileBWrite(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    int64_t value = bundle.getWaddrMulAttr().getInt();
    if (isVC4BundleMulPipeWriteActive(bundle) &&
        isVC4PhysicalRegfileWriteAddress(value)) {
      return value;
    }
    return std::nullopt;
  }
  if (auto ldi = llvm::dyn_cast<mlir::vc4::QPULDIOp>(op)) {
    int64_t value = ldi.getWaddrMulAttr().getInt();
    if (isVC4PhysicalRegfileWriteAddress(value))
      return value;
    return std::nullopt;
  }
  if (auto sema = llvm::dyn_cast<mlir::vc4::QPUSemaOp>(op)) {
    int64_t value = sema.getWaddrMulAttr().getInt();
    if (isVC4PhysicalRegfileWriteAddress(value))
      return value;
    return std::nullopt;
  }
  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    int64_t value = branch.getWaddrMulAttr().getInt();
    if (isVC4PhysicalRegfileWriteAddress(value))
      return value;
    return std::nullopt;
  }
  return std::nullopt;
}

static bool scheduledInstructionReadsPhysicalRegfileA(mlir::Operation *op,
                                                      int64_t value) {
  if (!isVC4PhysicalRegfileWriteAddress(value))
    return false;

  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op))
    return bundle.getRaddrAAttr().getInt() == value;

  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op))
    return branch.getRaddrAAttr().getInt() == value;

  return false;
}

static bool scheduledInstructionReadsPhysicalRegfileB(mlir::Operation *op,
                                                      int64_t value) {
  if (!isVC4PhysicalRegfileWriteAddress(value))
    return false;

  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    if (auto raddrB = bundle.getRaddrBAttr())
      return raddrB.getInt() == value;
  }

  return false;
}

static bool scheduledInstructionWritesR5(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    return (isVC4BundleAddPipeWriteActive(bundle) &&
            isVC4AccumulatorR5WriteAddress(bundle.getWaddrAddAttr().getInt())) ||
           (isVC4BundleMulPipeWriteActive(bundle) &&
            isVC4AccumulatorR5WriteAddress(bundle.getWaddrMulAttr().getInt()));
  }
  if (auto ldi = llvm::dyn_cast<mlir::vc4::QPULDIOp>(op)) {
    return isVC4AccumulatorR5WriteAddress(ldi.getWaddrAddAttr().getInt()) ||
           isVC4AccumulatorR5WriteAddress(ldi.getWaddrMulAttr().getInt());
  }
  if (auto sema = llvm::dyn_cast<mlir::vc4::QPUSemaOp>(op)) {
    return isVC4AccumulatorR5WriteAddress(sema.getWaddrAddAttr().getInt()) ||
           isVC4AccumulatorR5WriteAddress(sema.getWaddrMulAttr().getInt());
  }
  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    return isVC4AccumulatorR5WriteAddress(branch.getWaddrAddAttr().getInt()) ||
           isVC4AccumulatorR5WriteAddress(branch.getWaddrMulAttr().getInt());
  }
  return false;
}

static bool scheduledInstructionUsesRotateByR5SmallImm(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    if (bundle.getSig() != mlir::vc4::QPUSignal::small_imm)
      return false;
    if (auto smallImm = bundle.getSmallImmAttr())
      return smallImm.getInt() == 48;
  }
  return false;
}

static bool scheduledInstructionWritesSFU(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    return (isVC4BundleAddPipeWriteActive(bundle) &&
            isVC4SFUWriteAddress(bundle.getWaddrAddAttr().getInt())) ||
           (isVC4BundleMulPipeWriteActive(bundle) &&
            isVC4SFUWriteAddress(bundle.getWaddrMulAttr().getInt()));
  }
  if (auto ldi = llvm::dyn_cast<mlir::vc4::QPULDIOp>(op)) {
    return isVC4SFUWriteAddress(ldi.getWaddrAddAttr().getInt()) ||
           isVC4SFUWriteAddress(ldi.getWaddrMulAttr().getInt());
  }
  if (auto sema = llvm::dyn_cast<mlir::vc4::QPUSemaOp>(op)) {
    return isVC4SFUWriteAddress(sema.getWaddrAddAttr().getInt()) ||
           isVC4SFUWriteAddress(sema.getWaddrMulAttr().getInt());
  }
  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    return isVC4SFUWriteAddress(branch.getWaddrAddAttr().getInt()) ||
           isVC4SFUWriteAddress(branch.getWaddrMulAttr().getInt());
  }
  return false;
}

// In the current sink IR subset, an r4 read is visible only through the
// explicit QPU bundle muxes.
static bool scheduledInstructionReadsR4(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    return bundle.getAddA() == mlir::vc4::QPUMux::r4 ||
           bundle.getAddB() == mlir::vc4::QPUMux::r4 ||
           bundle.getMulA() == mlir::vc4::QPUMux::r4 ||
           bundle.getMulB() == mlir::vc4::QPUMux::r4;
  }
  return false;
}

// The currently representable sink-level r4-write-event subset is:
// - TMU receive signals encoded on vc4.qpu.bundle (ldtmu0 / ldtmu1)
// - any scheduled instruction that writes an SFU destination (52..55)
static bool scheduledInstructionTriggersR4WriteEventSubset(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    if (isVC4TMULoadSignal(bundle.getSig()))
      return true;
  }
  return scheduledInstructionWritesSFU(op);
}

static bool scheduledInstructionTouchesRegisterAddress14(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    if (isVC4RegisterAddress14(bundle.getRaddrAAttr().getInt()))
      return true;
    if (auto raddrB = bundle.getRaddrBAttr();
        raddrB && isVC4RegisterAddress14(raddrB.getInt()))
      return true;
    return (isVC4BundleAddPipeWriteActive(bundle) &&
            isVC4RegisterAddress14(bundle.getWaddrAddAttr().getInt())) ||
           (isVC4BundleMulPipeWriteActive(bundle) &&
            isVC4RegisterAddress14(bundle.getWaddrMulAttr().getInt()));
  }
  if (auto ldi = llvm::dyn_cast<mlir::vc4::QPULDIOp>(op)) {
    return isVC4RegisterAddress14(ldi.getWaddrAddAttr().getInt()) ||
           isVC4RegisterAddress14(ldi.getWaddrMulAttr().getInt());
  }
  if (auto sema = llvm::dyn_cast<mlir::vc4::QPUSemaOp>(op)) {
    return isVC4RegisterAddress14(sema.getWaddrAddAttr().getInt()) ||
           isVC4RegisterAddress14(sema.getWaddrMulAttr().getInt());
  }
  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    return isVC4RegisterAddress14(branch.getRaddrAAttr().getInt()) ||
           isVC4RegisterAddress14(branch.getWaddrAddAttr().getInt()) ||
           isVC4RegisterAddress14(branch.getWaddrMulAttr().getInt());
  }
  return false;
}

static bool scheduledInstructionAccessesUniformOrVPMVDW(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    if (isVC4RegisterAddress14(bundle.getRaddrAAttr().getInt()))
      return true;
    if (auto raddrB = bundle.getRaddrBAttr();
        raddrB && isVC4RegisterAddress14(raddrB.getInt()))
      return true;
    return (isVC4BundleAddPipeWriteActive(bundle) &&
            isVC4VPMOrDMAWriteAddress(bundle.getWaddrAddAttr().getInt())) ||
           (isVC4BundleMulPipeWriteActive(bundle) &&
            isVC4VPMOrDMAWriteAddress(bundle.getWaddrMulAttr().getInt()));
  }
  if (auto ldi = llvm::dyn_cast<mlir::vc4::QPULDIOp>(op)) {
    return isVC4VPMOrDMAWriteAddress(ldi.getWaddrAddAttr().getInt()) ||
           isVC4VPMOrDMAWriteAddress(ldi.getWaddrMulAttr().getInt());
  }
  if (auto sema = llvm::dyn_cast<mlir::vc4::QPUSemaOp>(op)) {
    return isVC4VPMOrDMAWriteAddress(sema.getWaddrAddAttr().getInt()) ||
           isVC4VPMOrDMAWriteAddress(sema.getWaddrMulAttr().getInt());
  }
  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    return isVC4RegisterAddress14(branch.getRaddrAAttr().getInt()) ||
           isVC4VPMOrDMAWriteAddress(branch.getWaddrAddAttr().getInt()) ||
           isVC4VPMOrDMAWriteAddress(branch.getWaddrMulAttr().getInt());
  }
  return false;
}

static mlir::LogicalResult appendVC4ScheduledInstructionStream(
    mlir::Operation *op, llvm::SmallVectorImpl<mlir::Operation *> &stream,
    llvm::StringRef verifierPassArg) {
  if (llvm::isa<mlir::vc4::QPUBundleOp, mlir::vc4::QPULDIOp,
                mlir::vc4::QPUSemaOp>(op)) {
    stream.push_back(op);
    return mlir::success();
  }

  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    stream.push_back(op);
    if (!branch.getDelaySlots().hasOneBlock()) {
      return op->emitOpError() << "must have exactly one delay-slot block for "
                               << verifierPassArg;
    }
    for (mlir::Operation &delaySlotOp : branch.getDelaySlots().front()) {
      if (mlir::failed(
              appendVC4ScheduledInstructionStream(&delaySlotOp, stream,
                                                 verifierPassArg))) {
        return mlir::failure();
      }
    }
    return mlir::success();
  }

  return op->emitOpError() << "is not a supported scheduled sink op for "
                           << verifierPassArg;
}

static mlir::LogicalResult collectVC4ScheduledInstructionStream(
    mlir::vc4::FuncOp func, llvm::StringRef verifierPassArg,
    llvm::SmallVectorImpl<mlir::Operation *> &stream) {
  if (!func.getBody().hasOneBlock()) {
    return func.emitOpError() << "scheduled qpu-domain functions checked by "
                              << verifierPassArg
                              << " must have a single top-level block";
  }

  for (mlir::Operation &op : func.getBody().front()) {
    if (mlir::failed(
            appendVC4ScheduledInstructionStream(&op, stream, verifierPassArg))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

struct VC4TestPrintEffectsPass
    : public mlir::PassWrapper<VC4TestPrintEffectsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VC4TestPrintEffectsPass)

  llvm::StringRef getArgument() const final { return "vc4-test-print-effects"; }
  llvm::StringRef getDescription() const final {
    return "Print VC4 MemoryEffectOpInterface effects";
  }

  void runOnOperation() final {
    getOperation()->walk([](mlir::Operation *op) {
      if (!op->getName().getStringRef().starts_with("vc4."))
        return;

      auto effectInterface = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(op);
      if (!effectInterface)
        return;

      llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
      effectInterface.getEffects(effects);
      if (effects.empty())
        return;

      llvm::outs() << op->getName().getStringRef() << ": ";
      for (size_t i = 0, e = effects.size(); i != e; ++i) {
        if (i)
          llvm::outs() << ", ";

        auto &effect = effects[i];
        llvm::StringRef effectName = "Effect";
        if (llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect()))
          effectName = "Read";
        else if (llvm::isa<mlir::MemoryEffects::Write>(effect.getEffect()))
          effectName = "Write";
        else if (llvm::isa<mlir::MemoryEffects::Allocate>(effect.getEffect()))
          effectName = "Allocate";
        else if (llvm::isa<mlir::MemoryEffects::Free>(effect.getEffect()))
          effectName = "Free";

        llvm::outs() << effectName << '<' << effect.getResource()->getName()
                     << '>';
      }
      llvm::outs() << '\n';
    });
  }
};

struct VC4VerifyEmitContractPass
    : public mlir::PassWrapper<VC4VerifyEmitContractPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VC4VerifyEmitContractPass)

  llvm::StringRef getArgument() const final {
    return "vc4-verify-emit-contract";
  }
  llvm::StringRef getDescription() const final {
    return "Verify which vc4.func operations are valid later emission inputs";
  }

  void runOnOperation() final {
    bool sawError = false;

    getOperation()->walk([&](mlir::vc4::FuncOp func) {
      if (sawError)
        return mlir::WalkResult::interrupt();

      if (func.isExternal())
        return mlir::WalkResult::advance();

      std::optional<mlir::vc4::ExecutionDomain> domain = func.getDomain();
      std::optional<mlir::vc4::FunctionForm> form = func.getForm();
      if (!domain || !form)
        return mlir::WalkResult::advance();

      if (*domain == mlir::vc4::ExecutionDomain::qpu &&
          *form == mlir::vc4::FunctionForm::structured) {
        func.emitOpError(
            "is not directly emittable: qasm emission later consumes only "
            "domain = #vc4.execution_domain<qpu>, "
            "form = #vc4.function_form<scheduled> functions; lower structured "
            "QPU ops such as uniforms/TMU/VPM/DMA/value-shape ops first");
        sawError = true;
        return mlir::WalkResult::interrupt();
      }

      if (*domain == mlir::vc4::ExecutionDomain::host &&
          *form == mlir::vc4::FunctionForm::scheduled) {
        func.emitOpError(
            "is not directly emittable: launcher generation later consumes "
            "only domain = #vc4.execution_domain<host>, "
            "form = #vc4.function_form<structured> functions");
        sawError = true;
        return mlir::WalkResult::interrupt();
      }

      if (*domain == mlir::vc4::ExecutionDomain::qpu &&
          *form == mlir::vc4::FunctionForm::scheduled) {
        func.getBody().walk([&](mlir::Operation *op) {
          if (isAllowedVC4QASMInputOp(*op))
            return mlir::WalkResult::advance();

          if (isVC4LauncherOrSystemOp(*op)) {
            op->emitOpError()
                << "is not a legal qasm-input op in "
                   "domain = #vc4.execution_domain<qpu>, "
                   "form = #vc4.function_form<scheduled> functions; "
                   "host/system ops belong in host structured functions";
            sawError = true;
            return mlir::WalkResult::interrupt();
          }

          if (isVC4StructuredFamilyOp(*op)) {
            op->emitOpError()
                << "is not a legal qasm-input op in "
                   "domain = #vc4.execution_domain<qpu>, "
                   "form = #vc4.function_form<scheduled> functions; lower "
                   "structured device ops to vc4.qpu.* sink ops first";
            sawError = true;
            return mlir::WalkResult::interrupt();
          }

          op->emitOpError()
              << "is not a supported scheduled sink op for qasm input; "
                 "expected only vc4.qpu.bundle, vc4.qpu.branch, "
                 "vc4.qpu.ldi, or vc4.qpu.sema";
          sawError = true;
          return mlir::WalkResult::interrupt();
        });
        return sawError ? mlir::WalkResult::interrupt()
                        : mlir::WalkResult::advance();
      }

      if (*domain == mlir::vc4::ExecutionDomain::host &&
          *form == mlir::vc4::FunctionForm::structured) {
        func.getBody().walk([&](mlir::Operation *op) {
          if (isAllowedVC4LauncherInputOp(*op))
            return mlir::WalkResult::advance();

          if (isVC4ScheduledSinkFamilyOp(*op)) {
            op->emitOpError()
                << "is not a legal launcher-input op in "
                   "domain = #vc4.execution_domain<host>, "
                   "form = #vc4.function_form<structured> functions; "
                   "scheduled sink ops belong in qpu scheduled functions";
            sawError = true;
            return mlir::WalkResult::interrupt();
          }

          if (isVC4StructuredFamilyOp(*op)) {
            op->emitOpError()
                << "is not a legal launcher-input op in "
                   "domain = #vc4.execution_domain<host>, "
                   "form = #vc4.function_form<structured> functions; "
                   "QPU/device ops belong in qpu functions";
            sawError = true;
            return mlir::WalkResult::interrupt();
          }

          op->emitOpError()
              << "is not a supported host/system launcher-input op; "
                 "expected only vc4.enqueue_qpu, vc4.reserve_qpu, "
                 "vc4.v3d.query, vc4.v3d.configure, vc4.async.wait, "
                 "vc4.cf.branch, or vc4.return";
          sawError = true;
          return mlir::WalkResult::interrupt();
        });
        return sawError ? mlir::WalkResult::interrupt()
                        : mlir::WalkResult::advance();
      }

      return mlir::WalkResult::advance();
    });

    if (sawError)
      signalPassFailure();
  }
};

// This pass checks only a narrow scheduled-hardware subset that cannot live in
// individual op verifiers:
// - thread-end signal instructions must not write physical regfile A/B
//   addresses 0..31
// - a thread-end window must not touch register-space address 14
// - the same window must not access the uniform read port (raddr_* = 14) or
//   VPM/VDR/VDW peripheral write destinations (waddr_* = 48..51)
// - last-thread-switch is only legal for threadable functions
//
// To make the instruction stream well-defined, the pass requires checked
// scheduled qpu-domain functions to have a single top-level block. The stream
// is then flattened in program order, with a vc4.qpu.branch contributing one
// instruction slot followed immediately by its explicit delay-slot ops in region
// order. The checked thread-end window is the signaling instruction plus the
// next two instruction slots in this flattened stream.
struct VC4VerifyScheduledHardwareRulesPass
    : public mlir::PassWrapper<VC4VerifyScheduledHardwareRulesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      VC4VerifyScheduledHardwareRulesPass)

  llvm::StringRef getArgument() const final {
    return "vc4-verify-scheduled-hardware-rules";
  }
  llvm::StringRef getDescription() const final {
    return "Verify a narrow subset of cross-instruction scheduled QPU hardware rules";
  }

  void runOnOperation() final {
    bool sawError = false;

    getOperation()->walk([&](mlir::vc4::FuncOp func) {
      if (sawError)
        return mlir::WalkResult::interrupt();
      if (func.isExternal() || !isVC4ScheduledQPUDomainFunction(func))
        return mlir::WalkResult::advance();

      llvm::SmallVector<mlir::Operation *> stream;
      if (mlir::failed(collectVC4ScheduledInstructionStream(
              func, "--vc4-verify-scheduled-hardware-rules", stream))) {
        sawError = true;
        return mlir::WalkResult::interrupt();
      }

      std::optional<mlir::vc4::ThreadingMode> threading = func.getThreading();
      for (size_t i = 0, e = stream.size(); i != e; ++i) {
        auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(stream[i]);
        if (!bundle)
          continue;

        if (bundle.getSig() == mlir::vc4::QPUSignal::last_thread_switch &&
            (!threading ||
             *threading != mlir::vc4::ThreadingMode::threadable)) {
          bundle.emitOpError()
              << "sig = #vc4.qpu_signal<last_thread_switch> is only legal in "
                 "functions with threading = "
                 "#vc4.threading_mode<threadable>";
          sawError = true;
          return mlir::WalkResult::interrupt();
        }

        if (!isVC4ThreadEndSignal(bundle.getSig()))
          continue;

        if (scheduledInstructionWritesPhysicalRegfile(bundle.getOperation())) {
          bundle.emitOpError()
              << "sig = #vc4.qpu_signal<"
              << mlir::vc4::stringifyQPUSignal(bundle.getSig())
              << "> must not write physical regfile A/B addresses 0..31";
          sawError = true;
          return mlir::WalkResult::interrupt();
        }

        for (size_t j = i, windowEnd = std::min(i + 3, e); j != windowEnd; ++j) {
          mlir::Operation *windowOp = stream[j];
          if (scheduledInstructionTouchesRegisterAddress14(windowOp)) {
            windowOp->emitOpError(
                "is in the thread-end hazard window and must not read or "
                "write regfile address 14");
            sawError = true;
            return mlir::WalkResult::interrupt();
          }
          if (scheduledInstructionAccessesUniformOrVPMVDW(windowOp)) {
            windowOp->emitOpError(
                "is in the thread-end hazard window and must not access "
                "uniforms or VPM/VDR/VDW register-space addresses (uniform "
                "read address 14, VPM/VDR/VDW write addresses 48..51)");
            sawError = true;
            return mlir::WalkResult::interrupt();
          }
        }
      }

      return mlir::WalkResult::advance();
    });

    if (sawError)
      signalPassFailure();
  }
};

// This verifier-only pass checks only a narrow adjacent-instruction
// scheduled-hardware subset that the current sink IR can represent directly:
// - regfile-A write -> next-instruction same regfile-A read
// - regfile-B write -> next-instruction same regfile-B read
// - r5 write -> next-instruction small_imm = 48 (rotate-by-r5)
// - SFU write -> next-two-instruction window must not:
//   - read r4 through a vc4.qpu.bundle source mux
//   - trigger another representable r4 write event, conservatively limited to
//     TMU receive signals (ldtmu0 / ldtmu1) or another SFU write
//
// As with --vc4-verify-scheduled-hardware-rules, the checked instruction
// stream is defined only for scheduled qpu-domain functions with a single
// top-level block. The stream is flattened in program order, with a
// vc4.qpu.branch contributing one instruction slot followed immediately by its
// explicit delay-slot ops in region order.
struct VC4VerifyScheduledAdjacentHazardsPass
    : public mlir::PassWrapper<VC4VerifyScheduledAdjacentHazardsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      VC4VerifyScheduledAdjacentHazardsPass)

  llvm::StringRef getArgument() const final {
    return "vc4-verify-scheduled-adjacent-hazards";
  }
  llvm::StringRef getDescription() const final {
    return "Verify a narrow subset of adjacent scheduled QPU hardware hazards";
  }

  void runOnOperation() final {
    bool sawError = false;

    getOperation()->walk([&](mlir::vc4::FuncOp func) {
      if (sawError)
        return mlir::WalkResult::interrupt();
      if (func.isExternal() || !isVC4ScheduledQPUDomainFunction(func))
        return mlir::WalkResult::advance();

      llvm::SmallVector<mlir::Operation *> stream;
      if (mlir::failed(collectVC4ScheduledInstructionStream(
              func, "--vc4-verify-scheduled-adjacent-hazards", stream))) {
        sawError = true;
        return mlir::WalkResult::interrupt();
      }

      for (size_t i = 0, e = stream.size(); i != e; ++i) {
        mlir::Operation *op = stream[i];
        mlir::Operation *next = i + 1 < e ? stream[i + 1] : nullptr;

        if (next) {
          if (std::optional<int64_t> aWrite =
                  scheduledInstructionPhysicalRegfileAWrite(op);
              aWrite && scheduledInstructionReadsPhysicalRegfileA(next, *aWrite)) {
            next->emitOpError()
                << "reads physical regfile-A address " << *aWrite
                << " written by the immediately previous scheduled instruction";
            sawError = true;
            return mlir::WalkResult::interrupt();
          }

          if (std::optional<int64_t> bWrite =
                  scheduledInstructionPhysicalRegfileBWrite(op);
              bWrite && scheduledInstructionReadsPhysicalRegfileB(next, *bWrite)) {
            next->emitOpError()
                << "reads physical regfile-B address " << *bWrite
                << " written by the immediately previous scheduled instruction";
            sawError = true;
            return mlir::WalkResult::interrupt();
          }

          if (scheduledInstructionWritesR5(op) &&
              scheduledInstructionUsesRotateByR5SmallImm(next)) {
            next->emitOpError(
                "uses small_imm = 48 (rotate-by-r5) immediately after an r5 "
                "write");
            sawError = true;
            return mlir::WalkResult::interrupt();
          }
        }

        if (!scheduledInstructionWritesSFU(op))
          continue;

        for (size_t j = i + 1, windowEnd = std::min(i + 3, e); j != windowEnd;
             ++j) {
          mlir::Operation *windowOp = stream[j];
          if (scheduledInstructionReadsR4(windowOp)) {
            windowOp->emitOpError(
                "is in the two-instruction SFU hazard window and must not "
                "read r4 through vc4.qpu.bundle source muxes");
            sawError = true;
            return mlir::WalkResult::interrupt();
          }
          if (scheduledInstructionTriggersR4WriteEventSubset(windowOp)) {
            windowOp->emitOpError(
                "is in the two-instruction SFU hazard window and must not "
                "trigger another representable r4 write event "
                "(ldtmu0/ldtmu1 or another SFU write)");
            sawError = true;
            return mlir::WalkResult::interrupt();
          }
        }
      }

      return mlir::WalkResult::advance();
    });

    if (sawError)
      signalPassFailure();
  }
};

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  mlir::PassRegistration<VC4TestPrintEffectsPass>();
  mlir::PassRegistration<VC4VerifyEmitContractPass>();
  mlir::PassRegistration<VC4VerifyScheduledHardwareRulesPass>();
  mlir::PassRegistration<VC4VerifyScheduledAdjacentHazardsPass>();

  mlir::DialectRegistry registry;
  registry.insert<mlir::vc4::VC4Dialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "VC4 modular optimizer driver\n", registry));
}
