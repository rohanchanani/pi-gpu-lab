//===- vc4-opt.cpp - VC4 standalone optimizer driver ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4/IR/VC4Dialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;
  registry.insert<mlir::vc4::VC4Dialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "VC4 modular optimizer driver\n", registry));
}
