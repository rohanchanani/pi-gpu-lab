//===- EmitBundle.h - VC4 artifact bundle emission -------------- C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VC4_TARGET_VC4_EMITBUNDLE_H
#define VC4_TARGET_VC4_EMITBUNDLE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace vc4 {

struct EmitBundleOptions {
 llvm::StringRef emitBundleDir;
};

LogicalResult emitVC4Bundle(::mlir::ModuleOp module,
                            const EmitBundleOptions &options);

} // namespace vc4
} // namespace mlir

#endif // VC4_TARGET_VC4_EMITBUNDLE_H
