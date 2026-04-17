#ifndef VC4_TARGET_KERNELMODEL_H
#define VC4_TARGET_KERNELMODEL_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "vc4/Dialect/VC4/VC4Ops.h"

#include <optional>
#include <string>

namespace mlir {
class Operation;

namespace gpu {
class GPUFuncOp;
} // namespace gpu

namespace vc4 {

enum class UniformSourceKind {
  PublicArgument,
  Builtin,
};

enum class KernelValueKind {
  UniformSlot,
  Builtin,
  LoopIndex,
  Temporary,
};

enum class BodyOperationKind {
  DmaLoad,
  StagedRead,
  FMul,
  FAdd,
  StagedWrite,
  DmaStore,
};

struct PublicArgumentInfo {
  unsigned index = 0;
  Type type;
};

struct UniformStreamEntry {
  unsigned slot = 0;
  Type type;
  UniformSourceKind sourceKind = UniformSourceKind::PublicArgument;
  unsigned publicArgumentIndex = 0;
  BuiltinKind builtinKind = BuiltinKind::qpu_id;
};

struct ExecutionMappingInfo {
  BuiltinKind workerIdBuiltin = BuiltinKind::qpu_id;
  BuiltinKind workerCountBuiltin = BuiltinKind::num_qpus;
  unsigned laneWidth = 0;
  unsigned baseMultiplier = 0;
  unsigned strideMultiplier = 0;
  unsigned upperBoundUniformSlot = 0;
  bool assumesUpperBoundMultipleOfLaneWidth = false;
};

struct KernelValueRef {
  KernelValueKind kind = KernelValueKind::Temporary;
  unsigned index = 0;
  BuiltinKind builtinKind = BuiltinKind::qpu_id;

  static KernelValueRef uniformSlot(unsigned slot);
  static KernelValueRef builtin(BuiltinKind builtinKind);
  static KernelValueRef loopIndex();
  static KernelValueRef temporary(unsigned index);
};

struct BodyOperationInfo {
  BodyOperationKind kind = BodyOperationKind::DmaLoad;
  std::optional<unsigned> memrefUniformSlot;
  std::optional<unsigned> stagingSlot;
  std::optional<KernelValueRef> offset;
  std::optional<KernelValueRef> source;
  std::optional<KernelValueRef> lhs;
  std::optional<KernelValueRef> rhs;
  std::optional<KernelValueRef> result;
};

struct LoweredKernelArtifactModel {
  std::string kernelName;
  llvm::SmallVector<PublicArgumentInfo> publicArguments;
  llvm::SmallVector<UniformStreamEntry> uniformStream;
  unsigned builtinSuffixStart = 0;
  ExecutionMappingInfo execution;
  // Ordered artifact-facing body stream extracted from lowered vc4 IR.
  // This is intentionally a thin handoff for later emission work, not a second
  // semantic IR.
  llvm::SmallVector<BodyOperationInfo> body;
};

FailureOr<LoweredKernelArtifactModel> extractLoweredKernelArtifactModel(Operation *op);
FailureOr<LoweredKernelArtifactModel> extractLoweredKernelArtifactModel(gpu::GPUFuncOp func);

LogicalResult printLoweredKernelArtifactModel(const LoweredKernelArtifactModel &model,
                                              llvm::raw_ostream &output);

void registerToVC4KernelModelTranslation();

} // namespace vc4
} // namespace mlir

#endif // VC4_TARGET_KERNELMODEL_H
