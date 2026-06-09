//===- vc4-triton-opt.cpp - Optional VC4 Triton frontend driver -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool is built only in the Triton-compatible optional lane.  It registers
// Triton's real `tt` dialect plus the VC4 standard value-layer dialects and the
// C++ TTIR -> VC4Value conversion pass.  The normal `vc4-opt` build must remain
// Triton-free.
//
//===----------------------------------------------------------------------===//

#include "vc4/Conversion/TritonToVC4Value/TritonToVC4Value.h"
#include "vc4/Dialect/VC4Value/IR/VC4ValueDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

int main(int argc, char **argv) {
  mlir::vc4::registerConvertTritonToVC4ValuePass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  mlir::vector::VectorDialect,
                  mlir::vc4value::VC4ValueDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "VC4 Triton frontend optimizer driver\n", registry));
}
