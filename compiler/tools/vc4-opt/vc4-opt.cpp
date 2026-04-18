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

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  mlir::PassRegistration<VC4TestPrintEffectsPass>();

  mlir::DialectRegistry registry;
  registry.insert<mlir::vc4::VC4Dialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "VC4 modular optimizer driver\n", registry));
}
