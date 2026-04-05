#ifndef PI_GPU_VC4_VC4DIALECT_H
#define PI_GPU_VC4_VC4DIALECT_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
}

namespace pi_gpu::vc4 {

void registerDialect(mlir::DialectRegistry &registry);
void loadDialect(mlir::MLIRContext &context);

}  // namespace pi_gpu::vc4

#endif
