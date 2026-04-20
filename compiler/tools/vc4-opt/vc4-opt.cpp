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

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  mlir::PassRegistration<VC4TestPrintEffectsPass>();
  mlir::PassRegistration<VC4VerifyEmitContractPass>();

  mlir::DialectRegistry registry;
  registry.insert<mlir::vc4::VC4Dialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "VC4 modular optimizer driver\n", registry));
}
