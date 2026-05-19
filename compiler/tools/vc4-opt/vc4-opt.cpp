//===- vc4-opt.cpp - VC4 standalone optimizer driver ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4/IR/VC4Ops.h"
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Dialect.h"
#include "vc4/Dialect/VC4/IR/VC4QPURegisterInfo.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"

namespace {

static bool isAllowedVC4QASMInputOp(mlir::Operation &op) {
  return llvm::isa<mlir::vc4::QPUBundleOp, mlir::vc4::QPUBranchOp,
                   mlir::vc4::QPULDIOp, mlir::vc4::QPUSemaOp,
                   mlir::vc4::QPUVPMVCDSetupOp,
                   mlir::vc4::QPUVPMVCDAddrOp,
                   mlir::vc4::QPUVPMVCDWaitOp>(op);
}

static bool isVC4ScheduledNonBranchOp(mlir::Operation &op) {
  return llvm::isa<mlir::vc4::QPUBundleOp, mlir::vc4::QPULDIOp,
                   mlir::vc4::QPUSemaOp, mlir::vc4::QPUVPMVCDSetupOp,
                   mlir::vc4::QPUVPMVCDAddrOp,
                   mlir::vc4::QPUVPMVCDWaitOp>(op);
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

static bool isVC4BundleAddPipeWriteActive(mlir::vc4::QPUBundleOp op) {
  return op.getOpAdd() != mlir::vc4::AddOpcode::nop;
}

static bool isVC4BundleMulPipeWriteActive(mlir::vc4::QPUBundleOp op) {
  return op.getOpMul() != mlir::vc4::MulOpcode::nop;
}

static bool isVC4ThreadEndSignal(mlir::vc4::QPUSignal signal) {
  return signal == mlir::vc4::QPUSignal::thrend;
}

static bool isVC4ThreadSwitchSignal(mlir::vc4::QPUSignal signal) {
  return signal == mlir::vc4::QPUSignal::thrsw ||
         signal == mlir::vc4::QPUSignal::last_thread_switch;
}

static bool isVC4TMULoadSignal(mlir::vc4::QPUSignal signal) {
  return signal == mlir::vc4::QPUSignal::ldtmu0 ||
         signal == mlir::vc4::QPUSignal::ldtmu1;
}

static bool isVC4SFUWriteAddress(int64_t value) {
  return mlir::vc4::isVC4QPUSFUWriteAddress(value);
}

static bool isVC4AccumulatorR5WriteAddress(int64_t value) {
  return mlir::vc4::isVC4QPUR5WriteAddress(value);
}

static bool isVC4AccumulatorR0ToR3WriteAddress(int64_t value) {
  return value >= 32 && value <= 35;
}

static bool isVC4AccumulatorR0ToR3Mux(mlir::vc4::QPUMux mux) {
  return mux == mlir::vc4::QPUMux::r0 || mux == mlir::vc4::QPUMux::r1 ||
         mux == mlir::vc4::QPUMux::r2 || mux == mlir::vc4::QPUMux::r3;
}

static std::optional<mlir::vc4::QPUMux>
getVC4AccumulatorR0ToR3MuxForWriteAddress(int64_t value) {
  switch (value) {
  case 32:
    return mlir::vc4::QPUMux::r0;
  case 33:
    return mlir::vc4::QPUMux::r1;
  case 34:
    return mlir::vc4::QPUMux::r2;
  case 35:
    return mlir::vc4::QPUMux::r3;
  default:
    return std::nullopt;
  }
}

static llvm::StringRef
getVC4AccumulatorR0ToR3Name(mlir::vc4::QPUMux mux) {
  switch (mux) {
  case mlir::vc4::QPUMux::r0:
    return "r0";
  case mlir::vc4::QPUMux::r1:
    return "r1";
  case mlir::vc4::QPUMux::r2:
    return "r2";
  case mlir::vc4::QPUMux::r3:
    return "r3";
  default:
    return "<invalid-accumulator>";
  }
}

static bool scheduledInstructionWritesPhysicalRegfile(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    return (isVC4BundleAddPipeWriteActive(bundle) &&
            mlir::vc4::isVC4QPUPhysicalRegfileAddress(
                bundle.getWaddrAddAttr().getInt())) ||
           (isVC4BundleMulPipeWriteActive(bundle) &&
            mlir::vc4::isVC4QPUPhysicalRegfileAddress(
                bundle.getWaddrMulAttr().getInt()));
  }
  if (auto ldi = llvm::dyn_cast<mlir::vc4::QPULDIOp>(op)) {
    return mlir::vc4::isVC4QPUPhysicalRegfileAddress(
               ldi.getWaddrAddAttr().getInt()) ||
           mlir::vc4::isVC4QPUPhysicalRegfileAddress(
               ldi.getWaddrMulAttr().getInt());
  }
  if (auto sema = llvm::dyn_cast<mlir::vc4::QPUSemaOp>(op)) {
    return mlir::vc4::isVC4QPUPhysicalRegfileAddress(
               sema.getWaddrAddAttr().getInt()) ||
           mlir::vc4::isVC4QPUPhysicalRegfileAddress(
               sema.getWaddrMulAttr().getInt());
  }
  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    return mlir::vc4::isVC4QPUPhysicalRegfileAddress(
               branch.getWaddrAddAttr().getInt()) ||
           mlir::vc4::isVC4QPUPhysicalRegfileAddress(
               branch.getWaddrMulAttr().getInt());
  }
  return false;
}

static std::optional<int64_t>
scheduledInstructionPhysicalRegfileAWrite(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    int64_t value = bundle.getWaddrAddAttr().getInt();
    if (isVC4BundleAddPipeWriteActive(bundle) &&
        mlir::vc4::isVC4QPUPhysicalRegfileAddress(value)) {
      return value;
    }
    return std::nullopt;
  }
  if (auto ldi = llvm::dyn_cast<mlir::vc4::QPULDIOp>(op)) {
    int64_t value = ldi.getWaddrAddAttr().getInt();
    if (mlir::vc4::isVC4QPUPhysicalRegfileAddress(value))
      return value;
    return std::nullopt;
  }
  if (auto sema = llvm::dyn_cast<mlir::vc4::QPUSemaOp>(op)) {
    int64_t value = sema.getWaddrAddAttr().getInt();
    if (mlir::vc4::isVC4QPUPhysicalRegfileAddress(value))
      return value;
    return std::nullopt;
  }
  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    int64_t value = branch.getWaddrAddAttr().getInt();
    if (mlir::vc4::isVC4QPUPhysicalRegfileAddress(value))
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
        mlir::vc4::isVC4QPUPhysicalRegfileAddress(value)) {
      return value;
    }
    return std::nullopt;
  }
  if (auto ldi = llvm::dyn_cast<mlir::vc4::QPULDIOp>(op)) {
    int64_t value = ldi.getWaddrMulAttr().getInt();
    if (mlir::vc4::isVC4QPUPhysicalRegfileAddress(value))
      return value;
    return std::nullopt;
  }
  if (auto sema = llvm::dyn_cast<mlir::vc4::QPUSemaOp>(op)) {
    int64_t value = sema.getWaddrMulAttr().getInt();
    if (mlir::vc4::isVC4QPUPhysicalRegfileAddress(value))
      return value;
    return std::nullopt;
  }
  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    int64_t value = branch.getWaddrMulAttr().getInt();
    if (mlir::vc4::isVC4QPUPhysicalRegfileAddress(value))
      return value;
    return std::nullopt;
  }
  return std::nullopt;
}

static bool scheduledInstructionReadsPhysicalRegfileA(mlir::Operation *op,
                                                      int64_t value) {
  if (!mlir::vc4::isVC4QPUPhysicalRegfileAddress(value))
    return false;

  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op))
    return bundle.getRaddrAAttr().getInt() == value;

  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op))
    return branch.getRaddrAAttr().getInt() == value;

  return false;
}

static bool scheduledInstructionReadsPhysicalRegfileB(mlir::Operation *op,
                                                      int64_t value) {
  if (!mlir::vc4::isVC4QPUPhysicalRegfileAddress(value))
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

static bool isVC4VectorRotateSmallImmSelector(int64_t value) {
  return value >= 48 && value <= 63;
}

static bool isVC4VectorRotateBundle(mlir::vc4::QPUBundleOp bundle) {
  if (bundle.getSig() != mlir::vc4::QPUSignal::small_imm)
    return false;
  if (auto smallImm = bundle.getSmallImmAttr())
    return isVC4VectorRotateSmallImmSelector(smallImm.getInt());
  return false;
}

static bool
scheduledBundleUsesAccumulatorR0ToR3(mlir::vc4::QPUBundleOp bundle,
                                     mlir::vc4::QPUMux mux) {
  if (!isVC4AccumulatorR0ToR3Mux(mux))
    return false;
  return bundle.getMulA() == mux || bundle.getMulB() == mux;
}

static std::optional<mlir::vc4::QPUMux>
scheduledInstructionAccumulatorR0ToR3AddWrite(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    int64_t value = bundle.getWaddrAddAttr().getInt();
    if (isVC4BundleAddPipeWriteActive(bundle) &&
        isVC4AccumulatorR0ToR3WriteAddress(value)) {
      return getVC4AccumulatorR0ToR3MuxForWriteAddress(value);
    }
    return std::nullopt;
  }
  if (auto ldi = llvm::dyn_cast<mlir::vc4::QPULDIOp>(op)) {
    return getVC4AccumulatorR0ToR3MuxForWriteAddress(
        ldi.getWaddrAddAttr().getInt());
  }
  if (auto sema = llvm::dyn_cast<mlir::vc4::QPUSemaOp>(op)) {
    return getVC4AccumulatorR0ToR3MuxForWriteAddress(
        sema.getWaddrAddAttr().getInt());
  }
  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    return getVC4AccumulatorR0ToR3MuxForWriteAddress(
        branch.getWaddrAddAttr().getInt());
  }
  return std::nullopt;
}

static std::optional<mlir::vc4::QPUMux>
scheduledInstructionAccumulatorR0ToR3MulWrite(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    int64_t value = bundle.getWaddrMulAttr().getInt();
    if (isVC4BundleMulPipeWriteActive(bundle) &&
        isVC4AccumulatorR0ToR3WriteAddress(value)) {
      return getVC4AccumulatorR0ToR3MuxForWriteAddress(value);
    }
    return std::nullopt;
  }
  if (auto ldi = llvm::dyn_cast<mlir::vc4::QPULDIOp>(op)) {
    return getVC4AccumulatorR0ToR3MuxForWriteAddress(
        ldi.getWaddrMulAttr().getInt());
  }
  if (auto sema = llvm::dyn_cast<mlir::vc4::QPUSemaOp>(op)) {
    return getVC4AccumulatorR0ToR3MuxForWriteAddress(
        sema.getWaddrMulAttr().getInt());
  }
  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    return getVC4AccumulatorR0ToR3MuxForWriteAddress(
        branch.getWaddrMulAttr().getInt());
  }
  return std::nullopt;
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

template <typename Predicate>
static bool scheduledInstructionReadsRegisterSpaceAddress(mlir::Operation *op,
                                                          Predicate predicate) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    if (predicate(bundle.getRaddrAAttr().getInt()))
      return true;
    if (auto raddrB = bundle.getRaddrBAttr())
      return predicate(raddrB.getInt());
    return false;
  }
  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op))
    return predicate(branch.getRaddrAAttr().getInt());
  return false;
}

template <typename Predicate>
static bool scheduledInstructionWritesRegisterSpaceAddress(
    mlir::Operation *op, Predicate predicate) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
    return (isVC4BundleAddPipeWriteActive(bundle) &&
            predicate(bundle.getWaddrAddAttr().getInt())) ||
           (isVC4BundleMulPipeWriteActive(bundle) &&
            predicate(bundle.getWaddrMulAttr().getInt()));
  }
  if (auto ldi = llvm::dyn_cast<mlir::vc4::QPULDIOp>(op)) {
    return predicate(ldi.getWaddrAddAttr().getInt()) ||
           predicate(ldi.getWaddrMulAttr().getInt());
  }
  if (auto sema = llvm::dyn_cast<mlir::vc4::QPUSemaOp>(op)) {
    return predicate(sema.getWaddrAddAttr().getInt()) ||
           predicate(sema.getWaddrMulAttr().getInt());
  }
  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    return predicate(branch.getWaddrAddAttr().getInt()) ||
           predicate(branch.getWaddrMulAttr().getInt());
  }
  return false;
}

template <typename Predicate>
static bool scheduledInstructionTouchesRegisterSpaceAddress(
    mlir::Operation *op, Predicate predicate) {
  return scheduledInstructionReadsRegisterSpaceAddress(op, predicate) ||
         scheduledInstructionWritesRegisterSpaceAddress(op, predicate);
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

static bool scheduledInstructionTouchesPhysicalRegfileAddress14(
    mlir::Operation *op) {
  return scheduledInstructionTouchesRegisterSpaceAddress(
      op, mlir::vc4::isVC4QPUThreadEndHazardPhysicalRegfileAddress);
}

static bool scheduledInstructionReadsUniform(mlir::Operation *op) {
  return scheduledInstructionReadsRegisterSpaceAddress(
      op, mlir::vc4::isVC4QPUUniformReadAddress);
}

static bool scheduledInstructionReadsVarying(mlir::Operation *op) {
  return scheduledInstructionReadsRegisterSpaceAddress(
      op, mlir::vc4::isVC4QPUVaryingReadAddress);
}

static bool scheduledInstructionTouchesVPMVDRVDWRegisterSpace(
    mlir::Operation *op) {
  return scheduledInstructionTouchesRegisterSpaceAddress(
      op, mlir::vc4::isVC4QPUVPMVDRVDWRegisterSpaceAddress);
}

static bool scheduledInstructionTouchesVPMVDRVDWControlRegisterSpace(
    mlir::Operation *op) {
  return scheduledInstructionTouchesRegisterSpaceAddress(
      op, mlir::vc4::isVC4QPUVPMDMAControlAddress);
}

static bool scheduledInstructionWritesUniformsAddress(mlir::Operation *op) {
  return scheduledInstructionWritesRegisterSpaceAddress(
      op, mlir::vc4::isVC4QPUUniformsAddress);
}

static bool scheduledInstructionWritesTMUNoswap(mlir::Operation *op) {
  return scheduledInstructionWritesRegisterSpaceAddress(
      op, mlir::vc4::isVC4QPUTMUNoswapAddress);
}

static bool scheduledInstructionWritesTMUParameter(mlir::Operation *op) {
  return scheduledInstructionWritesRegisterSpaceAddress(
      op, mlir::vc4::isVC4QPUTMUParameterWriteAddress);
}

static bool scheduledInstructionReadsMutexAcquire(mlir::Operation *op) {
  return scheduledInstructionReadsRegisterSpaceAddress(
      op, mlir::vc4::isVC4QPUMutexAddress);
}

static mlir::LogicalResult appendVC4ScheduledInstructionStream(
    mlir::Operation *op, llvm::SmallVectorImpl<mlir::Operation *> &stream,
    llvm::StringRef verifierPassArg) {
  if (llvm::isa<mlir::vc4::QPUBundleOp, mlir::vc4::QPULDIOp,
                mlir::vc4::QPUSemaOp, mlir::vc4::QPUVPMVCDSetupOp,
                mlir::vc4::QPUVPMVCDAddrOp,
                mlir::vc4::QPUVPMVCDWaitOp>(op)) {
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

static mlir::InFlightDiagnostic
emitInvalidQASMEpilogueDiag(mlir::vc4::FuncOp func) {
  return func.emitOpError(
      "is not directly emittable: qasm input requires an explicit thrend plus "
      "two delay-slot instructions at the end of the flattened scheduled "
      "instruction stream");
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

      if (*domain == mlir::vc4::ExecutionDomain::host &&
          *form == mlir::vc4::FunctionForm::scheduled) {
        func.emitOpError(
            "is not directly emittable: scheduled VC4 functions require "
            "domain = #vc4.execution_domain<qpu>");
        sawError = true;
        return mlir::WalkResult::interrupt();
      }

      if (*domain == mlir::vc4::ExecutionDomain::qpu &&
          *form == mlir::vc4::FunctionForm::scheduled) {
        func.getBody().walk([&](mlir::Operation *op) {
          if (isAllowedVC4QASMInputOp(*op))
            return mlir::WalkResult::advance();

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
        if (sawError)
          return mlir::WalkResult::interrupt();

        llvm::SmallVector<mlir::Operation *> stream;
        if (mlir::failed(collectVC4ScheduledInstructionStream(
                func, "--vc4-verify-emit-contract", stream))) {
          sawError = true;
          return mlir::WalkResult::interrupt();
        }

        if (stream.size() < 3) {
          emitInvalidQASMEpilogueDiag(func)
              << "; found only " << stream.size()
              << " scheduled instruction slot(s)";
          sawError = true;
          return mlir::WalkResult::interrupt();
        }

        size_t epilogueStart = stream.size() - 3;
        auto finalThreadEnd =
            llvm::dyn_cast<mlir::vc4::QPUBundleOp>(stream[epilogueStart]);
        if (!finalThreadEnd || !isVC4ThreadEndSignal(finalThreadEnd.getSig())) {
          emitInvalidQASMEpilogueDiag(func)
              << "; slot N-3 must be a vc4.qpu.bundle with sig = "
                 "#vc4.qpu_signal<thrend>";
          sawError = true;
          return mlir::WalkResult::interrupt();
        }

        for (size_t i = 0; i != epilogueStart; ++i) {
          if (auto earlierThreadEnd =
                  llvm::dyn_cast<mlir::vc4::QPUBundleOp>(stream[i]);
              earlierThreadEnd &&
              isVC4ThreadEndSignal(earlierThreadEnd.getSig())) {
            emitInvalidQASMEpilogueDiag(func)
                << "; found an earlier vc4.qpu.bundle with sig = "
                   "#vc4.qpu_signal<thrend> before slot N-3";
            sawError = true;
            return mlir::WalkResult::interrupt();
          }
        }

        if (!isVC4ScheduledNonBranchOp(*stream[epilogueStart + 1])) {
          emitInvalidQASMEpilogueDiag(func)
              << "; slot N-2 must be a non-branch scheduled op "
                 "(vc4.qpu.bundle, vc4.qpu.ldi, or vc4.qpu.sema)";
          sawError = true;
          return mlir::WalkResult::interrupt();
        }

        if (!isVC4ScheduledNonBranchOp(*stream[epilogueStart + 2])) {
          emitInvalidQASMEpilogueDiag(func)
              << "; slot N-1 must be a non-branch scheduled op "
                 "(vc4.qpu.bundle, vc4.qpu.ldi, or vc4.qpu.sema)";
          sawError = true;
          return mlir::WalkResult::interrupt();
        }

        if (auto trailingThreadEnd =
                llvm::dyn_cast<mlir::vc4::QPUBundleOp>(stream[epilogueStart + 1]);
            (trailingThreadEnd &&
             isVC4ThreadEndSignal(trailingThreadEnd.getSig())) ||
            (llvm::dyn_cast<mlir::vc4::QPUBundleOp>(stream[epilogueStart + 2]) &&
             isVC4ThreadEndSignal(
                 llvm::cast<mlir::vc4::QPUBundleOp>(stream[epilogueStart + 2])
                     .getSig()))) {
          emitInvalidQASMEpilogueDiag(func)
              << "; only slot N-3 may carry sig = "
                 "#vc4.qpu_signal<thrend>; slots N-2 and N-1 must be "
                 "non-branch scheduled ops without another thread-end signal";
          sawError = true;
          return mlir::WalkResult::interrupt();
        }

        return mlir::WalkResult::advance();
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
// - a thread-end window must not touch physical regfile address 14
// - the same window must not read the uniform register-space address 32
// - the same window must not read the varying register-space address 35
// - the same window must not access VPM/VDR/VDW register-space addresses
//   48..50
// - thread-switch signals are only legal for threadable functions
// - the final thread-switch signal in the flattened stream must be
//   last_thread_switch
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
      std::optional<size_t> lastThreadSwitchIndex;
      for (size_t i = 0, e = stream.size(); i != e; ++i) {
        auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(stream[i]);
        if (!bundle)
          continue;

        if (isVC4ThreadSwitchSignal(bundle.getSig())) {
          if (!threading ||
              *threading != mlir::vc4::ThreadingMode::threadable) {
            if (bundle.getSig() == mlir::vc4::QPUSignal::thrsw) {
              bundle.emitOpError()
                  << "sig = #vc4.qpu_signal<thrsw> is only legal in "
                     "functions with threading = "
                     "#vc4.threading_mode<threadable>";
            } else {
              bundle.emitOpError()
                  << "sig = #vc4.qpu_signal<last_thread_switch> is only legal "
                     "in functions with threading = "
                     "#vc4.threading_mode<threadable>";
            }
            sawError = true;
            return mlir::WalkResult::interrupt();
          }

          if (i + 2 >= e) {
            if (bundle.getSig() == mlir::vc4::QPUSignal::thrsw) {
              bundle.emitOpError()
                  << "sig = #vc4.qpu_signal<thrsw> requires two following "
                     "delay-slot instructions in the flattened scheduled "
                     "instruction stream";
            } else {
              bundle.emitOpError()
                  << "sig = #vc4.qpu_signal<last_thread_switch> requires two "
                     "following delay-slot instructions in the flattened "
                     "scheduled instruction stream";
            }
            sawError = true;
            return mlir::WalkResult::interrupt();
          }

          lastThreadSwitchIndex = i;
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
          if (scheduledInstructionTouchesPhysicalRegfileAddress14(windowOp)) {
            windowOp->emitOpError()
                << "is in the thread-end hazard window and must not read or "
                   "write physical regfile address "
                << mlir::vc4::kVC4QPUThreadEndHazardPhysicalRegfileAddr;
            sawError = true;
            return mlir::WalkResult::interrupt();
          }
          if (scheduledInstructionReadsUniform(windowOp)) {
            windowOp->emitOpError()
                << "is in the thread-end hazard window and must not read "
                   "uniform register-space address "
                << mlir::vc4::kVC4QPUUniformRead;
            sawError = true;
            return mlir::WalkResult::interrupt();
          }
          if (scheduledInstructionReadsVarying(windowOp)) {
            windowOp->emitOpError()
                << "is in the thread-end hazard window and must not read "
                   "varying register-space address "
                << mlir::vc4::kVC4QPUVaryingRead;
            sawError = true;
            return mlir::WalkResult::interrupt();
          }
          if (scheduledInstructionTouchesVPMVDRVDWRegisterSpace(windowOp)) {
            windowOp->emitOpError()
                << "is in the thread-end hazard window and must not access "
                   "VPM/VDR/VDW register-space addresses "
                << mlir::vc4::kVC4QPUVPMVDRVDWMin << ".."
                << mlir::vc4::kVC4QPUVPMVDRVDWMax;
            sawError = true;
            return mlir::WalkResult::interrupt();
          }
        }
      }

      if (lastThreadSwitchIndex) {
        for (size_t i = 0; i != *lastThreadSwitchIndex; ++i) {
          auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(stream[i]);
          if (bundle &&
              bundle.getSig() == mlir::vc4::QPUSignal::last_thread_switch) {
            bundle.emitOpError()
                << "sig = #vc4.qpu_signal<last_thread_switch> must be the "
                   "final thread-switch signal in the flattened scheduled "
                   "instruction stream";
            sawError = true;
            return mlir::WalkResult::interrupt();
          }
        }

        auto finalThreadSwitch =
            llvm::cast<mlir::vc4::QPUBundleOp>(stream[*lastThreadSwitchIndex]);
        if (finalThreadSwitch.getSig() == mlir::vc4::QPUSignal::thrsw) {
          finalThreadSwitch.emitOpError()
              << "the final thread-switch signal in the flattened scheduled "
                 "instruction stream must be "
                 "#vc4.qpu_signal<last_thread_switch>";
          sawError = true;
          return mlir::WalkResult::interrupt();
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
// - accumulator r0..r3 write -> next-instruction vector rotate using that
//   same accumulator on the MUL side
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

          if (auto nextBundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(next);
              nextBundle && isVC4VectorRotateBundle(nextBundle)) {
            auto emitVectorRotateHazard =
                [&](std::optional<mlir::vc4::QPUMux> writtenAccumulator) {
                  if (!writtenAccumulator ||
                      !scheduledBundleUsesAccumulatorR0ToR3(
                          nextBundle, *writtenAccumulator)) {
                    return false;
                  }
                  nextBundle.emitOpError()
                      << "does a vector rotate immediately after the previous "
                         "scheduled instruction wrote accumulator "
                      << getVC4AccumulatorR0ToR3Name(*writtenAccumulator);
                  sawError = true;
                  return true;
                };

            if (emitVectorRotateHazard(
                    scheduledInstructionAccumulatorR0ToR3AddWrite(op)) ||
                emitVectorRotateHazard(
                    scheduledInstructionAccumulatorR0ToR3MulWrite(op))) {
              return mlir::WalkResult::interrupt();
            }
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

// This verifier-only pass checks only two sink-level IO spacing rules that are
// directly relevant to compute kernels in the current scheduled subset:
// - a write to UNIFORMS_ADDRESS (40) must not be followed within the next two
//   instruction slots by a uniform read (raddr = 32)
// - after a write to TMU_NOSWAP (36), the first later TMU parameter write
//   (56..63) must be at least three instruction slots later
//
// As with the other scheduled verifiers, the checked instruction stream is
// defined only for scheduled qpu-domain functions with a single top-level
// block. The stream is flattened in program order, with a vc4.qpu.branch
// contributing one instruction slot followed immediately by its explicit
// delay-slot ops in region order.
struct VC4VerifyScheduledIOSpacingPass
    : public mlir::PassWrapper<VC4VerifyScheduledIOSpacingPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VC4VerifyScheduledIOSpacingPass)

  llvm::StringRef getArgument() const final {
    return "vc4-verify-scheduled-io-spacing";
  }
  llvm::StringRef getDescription() const final {
    return "Verify a narrow subset of scheduled QPU IO spacing rules";
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
              func, "--vc4-verify-scheduled-io-spacing", stream))) {
        sawError = true;
        return mlir::WalkResult::interrupt();
      }

      for (size_t i = 0, e = stream.size(); i != e; ++i) {
        mlir::Operation *op = stream[i];

        if (scheduledInstructionWritesUniformsAddress(op)) {
          for (size_t j = i + 1, windowEnd = std::min(i + 3, e); j != windowEnd;
               ++j) {
            mlir::Operation *windowOp = stream[j];
            if (!scheduledInstructionReadsUniform(windowOp))
              continue;
            windowOp->emitOpError()
                << "reads uniform register-space address "
                << mlir::vc4::kVC4QPUUniformRead
                << " within two instruction slots after a write to "
                   "UNIFORMS_ADDRESS "
                << mlir::vc4::kVC4QPUUniformsAddress;
            sawError = true;
            return mlir::WalkResult::interrupt();
          }
        }

        if (!scheduledInstructionWritesTMUNoswap(op))
          continue;

        for (size_t j = i + 1; j != e; ++j) {
          mlir::Operation *laterOp = stream[j];
          if (!scheduledInstructionWritesTMUParameter(laterOp))
            continue;
          size_t distance = j - i;
          if (distance < 3) {
            laterOp->emitOpError()
                << "writes TMU parameter register-space addresses "
                << mlir::vc4::kVC4QPUTMUParameterWriteMin << ".."
                << mlir::vc4::kVC4QPUTMUParameterWriteMax << " only "
                << distance
                << " instruction slot(s) after a write to TMU_NOSWAP "
                << mlir::vc4::kVC4QPUTMUNoswap
                << "; the first later TMU parameter write must be at least "
                   "three instruction slots later";
            sawError = true;
            return mlir::WalkResult::interrupt();
          }
          break;
        }
      }

      return mlir::WalkResult::advance();
    });

    if (sawError)
      signalPassFailure();
  }
};

// This verifier-only pass checks a conservative single-slot subset of
// closely-coupled peripheral accesses that are directly representable in the
// current compute-focused scheduled sink IR:
// - TMU read signal on vc4.qpu.bundle (ldtmu0 / ldtmu1)
// - TMU parameter write (56..63)
// - SFU write (52..55)
// - mutex acquire read through a representable sink instruction (raddr = 51)
// - semaphore access (vc4.qpu.sema)
// - VPM / VDR / VDW control register-space access (49..50)
//
// Any flattened scheduled instruction slot that encodes more than one of those
// access categories is rejected.
struct VC4VerifyScheduledPeripheralAccessesPass
    : public mlir::PassWrapper<VC4VerifyScheduledPeripheralAccessesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      VC4VerifyScheduledPeripheralAccessesPass)

  llvm::StringRef getArgument() const final {
    return "vc4-verify-scheduled-peripheral-accesses";
  }
  llvm::StringRef getDescription() const final {
    return "Verify conservative single-slot scheduled QPU peripheral-access combinations";
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
              func, "--vc4-verify-scheduled-peripheral-accesses", stream))) {
        sawError = true;
        return mlir::WalkResult::interrupt();
      }

      for (mlir::Operation *op : stream) {
        bool hasTMUReadSignal = false;
        if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op))
          hasTMUReadSignal = isVC4TMULoadSignal(bundle.getSig());

        bool hasTMUParameterWrite = scheduledInstructionWritesTMUParameter(op);
        bool hasSFUWrite = scheduledInstructionWritesSFU(op);
        bool hasMutexAcquireRead = scheduledInstructionReadsMutexAcquire(op);
        bool hasSemaphoreAccess = llvm::isa<mlir::vc4::QPUSemaOp>(op);
        bool hasVPMVDRVDWControlAccess =
            scheduledInstructionTouchesVPMVDRVDWControlRegisterSpace(op);

        unsigned accessCount = static_cast<unsigned>(hasTMUReadSignal) +
                               static_cast<unsigned>(hasTMUParameterWrite) +
                               static_cast<unsigned>(hasSFUWrite) +
                               static_cast<unsigned>(hasMutexAcquireRead) +
                               static_cast<unsigned>(hasSemaphoreAccess) +
                               static_cast<unsigned>(
                                   hasVPMVDRVDWControlAccess);
        if (accessCount <= 1)
          continue;

        auto diag = op->emitOpError(
            "encodes more than one closely-coupled peripheral access in a "
            "single scheduled instruction slot (");
        bool firstCategory = true;
        auto appendCategory = [&](llvm::StringRef category) {
          if (!firstCategory)
            diag << ", ";
          diag << category;
          firstCategory = false;
        };

        if (hasTMUReadSignal)
          appendCategory("TMU read signal");
        if (hasTMUParameterWrite)
          appendCategory("TMU parameter write");
        if (hasSFUWrite)
          appendCategory("SFU write");
        if (hasMutexAcquireRead)
          appendCategory("mutex acquire read");
        if (hasSemaphoreAccess)
          appendCategory("semaphore access");
        if (hasVPMVDRVDWControlAccess)
          appendCategory("VPM/VDR/VDW control register-space access");
        diag << ")";

        sawError = true;
        return mlir::WalkResult::interrupt();
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
  mlir::PassRegistration<VC4VerifyScheduledIOSpacingPass>();
  mlir::PassRegistration<VC4VerifyScheduledPeripheralAccessesPass>();

  mlir::DialectRegistry registry;
  registry.insert<mlir::vc4::VC4Dialect, mlir::ssavc4::SSAVC4Dialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "VC4 modular optimizer driver\n", registry));
}
