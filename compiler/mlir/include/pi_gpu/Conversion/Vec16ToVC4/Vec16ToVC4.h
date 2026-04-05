#ifndef PI_GPU_CONVERSION_VEC16TOVC4_VEC16TOVC4_H
#define PI_GPU_CONVERSION_VEC16TOVC4_VEC16TOVC4_H

#include <memory>

namespace mlir {
class Pass;
class PassRegistry;
}

namespace pi_gpu {

std::unique_ptr<mlir::Pass> createConvertVec16ToVC4Pass();
void registerConvertVec16ToVC4Pass();

}  // namespace pi_gpu

#endif
