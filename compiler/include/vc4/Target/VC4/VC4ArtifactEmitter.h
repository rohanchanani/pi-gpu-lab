//===- VC4ArtifactEmitter.h - VC4 codegen artifact bundle API ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VC4_TARGET_VC4_VC4ARTIFACTEMITTER_H
#define VC4_TARGET_VC4_VC4ARTIFACTEMITTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace mlir::vc4 {

struct VC4ArtifactKernelInfo {
  std::string symbolName;
  std::string publicName;
  int64_t uniformWordsPerQPU = 0;
  uint64_t spillFrameBytes = 0;
  uint64_t spillFrameStrideBytes = 0;
  uint64_t spillFrameCount = 0;
  uint64_t spillArenaBytes = 0;
  unsigned scheduledOpCount = 0;
};

/// Validate the final scheduled VC4 input boundary and emit a skeleton artifact
/// bundle rooted at `bundleDir`.
LogicalResult emitVC4ArtifactBundle(mlir::ModuleOp module,
                                    llvm::StringRef bundleDir);

} // namespace mlir::vc4

#endif // VC4_TARGET_VC4_VC4ARTIFACTEMITTER_H
