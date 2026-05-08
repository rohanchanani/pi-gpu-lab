//===- vc4-codegen.cpp - VC4 artifact bundle emitter driver ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Target/VC4/VC4ArtifactEmitter.h"

#include "vc4/Dialect/VC4/IR/VC4Ops.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input.mlir>"), llvm::cl::Required);

static llvm::cl::opt<std::string> emitBundleDir(
    "emit-bundle", llvm::cl::desc("Directory for generated VC4 artifacts"),
    llvm::cl::value_desc("directory"));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "VC4 codegen artifact bundle emitter\n");

  if (emitBundleDir.getValue().empty()) {
    llvm::errs() << "vc4-codegen: error: --emit-bundle is required\n";
    return 1;
  }

  DialectRegistry registry;
  registry.insert<mlir::vc4::VC4Dialect>();

  MLIRContext context(registry);
  context.loadDialect<mlir::vc4::VC4Dialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(inputFilename.getValue(), &context);
  if (!module) {
    llvm::errs() << "vc4-codegen: error: failed to parse '"
                 << inputFilename.getValue() << "'\n";
    return 1;
  }

  if (failed(verify(module.get().getOperation()))) {
    llvm::errs() << "vc4-codegen: error: input verification failed\n";
    return 1;
  }

  if (failed(mlir::vc4::emitVC4ArtifactBundle(module.get(),
                                             emitBundleDir.getValue())))
    return 1;

  return 0;
}
