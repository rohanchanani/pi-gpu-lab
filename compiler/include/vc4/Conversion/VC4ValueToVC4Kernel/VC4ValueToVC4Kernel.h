//===- VC4ValueToVC4Kernel.h - VC4 value to VC4Kernel passes ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VC4_CONVERSION_VC4VALUETOVC4KERNEL_VC4VALUETOVC4KERNEL_H
#define VC4_CONVERSION_VC4VALUETOVC4KERNEL_VC4VALUETOVC4KERNEL_H

#include <memory>

namespace mlir {
class Pass;

namespace vc4 {

/// Creates the Phase 5 value-surface to VC4Kernel conversion pass.
///
/// This pass lowers the first handwritten value-layer executable subset into
/// the locked VC4Kernel target-kernel planning dialect.  It is intentionally
/// narrower than the full value-surface contract: Phase 5 accepts only
/// straight-line `func.func` kernels marked `vc4value.kernel`, 1-D i32/f32
/// global memrefs, vector<16> elementwise operations, and contiguous
/// transfer_read/transfer_write patterns.  Staged value-surface forms must
/// produce deterministic diagnostics rather than being silently accepted.
std::unique_ptr<Pass> createConvertVC4ValueToVC4KernelPass();

/// Registers --convert-vc4-value-to-vc4kernel.
void registerConvertVC4ValueToVC4KernelPass();

} // namespace vc4
} // namespace mlir

#endif // VC4_CONVERSION_VC4VALUETOVC4KERNEL_VC4VALUETOVC4KERNEL_H
