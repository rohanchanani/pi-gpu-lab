#ifndef VC4_TARGET_VC4ASM_VC4ASM_H
#define VC4_TARGET_VC4ASM_VC4ASM_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class DialectRegistry;
class Operation;

namespace vc4 {

struct LoweredKernelArtifactModel;

LogicalResult emitVC4QASM(Operation *op, llvm::raw_ostream &output);
LogicalResult emitVC4QASM(const LoweredKernelArtifactModel &model,
                          llvm::raw_ostream &output);
LogicalResult emitVC4LauncherHeader(Operation *op, llvm::raw_ostream &output);
LogicalResult emitVC4LauncherHeader(const LoweredKernelArtifactModel &model,
                                    llvm::raw_ostream &output);
LogicalResult emitVC4LauncherSource(Operation *op, llvm::raw_ostream &output);
LogicalResult emitVC4LauncherSource(const LoweredKernelArtifactModel &model,
                                    llvm::raw_ostream &output);
LogicalResult emitVC4Artifacts(Operation *op, llvm::StringRef outputDir);

void registerToVC4AsmTranslation();
void registerToVC4QASMTranslation();
void registerToVC4LauncherHeaderTranslation();
void registerToVC4LauncherSourceTranslation();

} // namespace vc4
} // namespace mlir

#endif // VC4_TARGET_VC4ASM_VC4ASM_H
