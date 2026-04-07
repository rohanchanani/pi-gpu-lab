#ifndef VC4_DIALECT_VC4_VC4OPS_H
#define VC4_DIALECT_VC4_VC4OPS_H

#include <optional>

#include "llvm/ADT/StringRef.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "vc4/Dialect/VC4/VC4Dialect.h"

namespace mlir::vc4 {

enum class BuiltinKind {
  qpu_id,
  num_qpus,
};

llvm::StringRef stringifyBuiltinKind(BuiltinKind kind);
std::optional<BuiltinKind> symbolizeBuiltinKind(llvm::StringRef kind);

struct UniformResource : public SideEffects::Resource::Base<UniformResource> {
  StringRef getName() final { return "VC4::Uniforms"; }
};

struct VPMResource : public SideEffects::Resource::Base<VPMResource> {
  StringRef getName() final { return "VC4::VPM"; }
};

struct GlobalMemoryResource
    : public SideEffects::Resource::Base<GlobalMemoryResource> {
  StringRef getName() final { return "VC4::GlobalMemory"; }
};

} // namespace mlir::vc4

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4/VC4Ops.h.inc"

#endif // VC4_DIALECT_VC4_VC4OPS_H
