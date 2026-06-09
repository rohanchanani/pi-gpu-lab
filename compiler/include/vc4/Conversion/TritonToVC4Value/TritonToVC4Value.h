//===- TritonToVC4Value.h - TTIR to VC4 value lowering ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the optional Triton TTIR -> VC4 value-layer conversion.
// The conversion consumes real MLIR parsed with Triton's `tt` dialect registered
// and emits only the standard VC4 value surface:
//
//   func + vc4value + vector + memref + arith + math + scf/cf
//
// It must never emit vc4kernel/ssavc4/scheduled-vc4 directly.  The required
// producer path remains:
//
//   TTIR -> VC4 value -> VC4Kernel -> SSAVC4 -> scheduled VC4 -> hardware
//
// The pass itself intentionally avoids depending on Triton generated C++ op
// classes.  It operates on parsed MLIR Operations structurally, by operation
// name, SSA use-defs, attributes, and types.  The optional frontend tool is
// responsible for registering Triton's dialect so the input TTIR parses as real
// typed MLIR rather than as text templates.
//
//===----------------------------------------------------------------------===//

#ifndef VC4_CONVERSION_TRITONTOVC4VALUE_TRITONTOVC4VALUE_H
#define VC4_CONVERSION_TRITONTOVC4VALUE_TRITONTOVC4VALUE_H

#include <memory>

namespace mlir {
class Pass;

namespace vc4 {

std::unique_ptr<Pass> createConvertTritonToVC4ValuePass();
void registerConvertTritonToVC4ValuePass();

} // namespace vc4
} // namespace mlir

#endif // VC4_CONVERSION_TRITONTOVC4VALUE_TRITONTOVC4VALUE_H
