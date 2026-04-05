#include "pi_gpu/VC4/VC4Dialect.h"

namespace pi_gpu::vc4 {

namespace {

class VC4Dialect;

class VC4UniformOp;
class VC4DmaLoadVPMOp;
class VC4VPMReadOp;
class VC4BinOp;
class VC4DmaStoreVPMOp;

class VC4Dialect {
 public:
  static constexpr const char *getDialectNamespace() { return "vc4"; }
};

class VC4UniformOp {
 public:
  static constexpr const char *getOperationName() { return "vc4.uniform"; }
};

class VC4DmaLoadVPMOp {
 public:
  static constexpr const char *getOperationName() {
    return "vc4.dma_load_vpm";
  }
};

class VC4VPMReadOp {
 public:
  static constexpr const char *getOperationName() { return "vc4.vpm_read"; }
};

class VC4BinOp {
 public:
  static constexpr const char *getOperationName() { return "vc4.bin"; }
};

class VC4DmaStoreVPMOp {
 public:
  static constexpr const char *getOperationName() {
    return "vc4.dma_store_vpm";
  }
};

}  // namespace

void registerDialect(mlir::DialectRegistry &registry) {
  (void)registry;
  // Placeholder until the dialect is wired to real MLIR generated classes.
}

void loadDialect(mlir::MLIRContext &context) { (void)context; }

}  // namespace pi_gpu::vc4
