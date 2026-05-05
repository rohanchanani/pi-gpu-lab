//===- vc4-codegen.cpp - VC4 artifact codegen driver ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4/IR/VC4Ops.h"
#include "vc4/Target/VC4/EmitBundle.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

int main(int argc, char **argv) {
 llvm::InitLLVM y(argc, argv);

 llvm::cl::opt<std::string> inputFilename(
 llvm::cl::Positional, llvm::cl::desc("<input mlir>"),
 llvm::cl::Required);
 llvm::cl::opt<std::string> emitBundleDir(
 "emit-bundle", llvm::cl::desc("Directory to receive VC4 bundle artifacts"),
 llvm::cl::value_desc("dir"), llvm::cl::Required);

 llvm::cl::ParseCommandLineOptions(argc, argv,
 "VC4 artifact codegen driver\n");

 mlir::DialectRegistry registry;
 registry.insert<mlir::vc4::VC4Dialect>();

 mlir::MLIRContext context(registry);
 context.allowUnregisteredDialects(false);

 std::string errorMessage;
 std::unique_ptr<llvm::MemoryBuffer> input =
 mlir::openInputFile(inputFilename, &errorMessage);
 if (!input) {
 llvm::errs() << errorMessage << '\n';
 return 1;
 }

 llvm::SourceMgr sourceMgr;
 sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
 mlir::SourceMgrDiagnosticHandler diagnosticHandler(sourceMgr, &context);

 mlir::ParserConfig parserConfig(&context, true);
 mlir::OwningOpRef<mlir::ModuleOp> module =
 mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, parserConfig);
 if (!module)
 return 1;

 if (mlir::failed(mlir::verify(*module)))
 return 1;

 mlir::vc4::EmitBundleOptions options;
 options.emitBundleDir = emitBundleDir;

 if (mlir::failed(mlir::vc4::emitVC4Bundle(*module, options)))
 return 1;

 return 0;
}
