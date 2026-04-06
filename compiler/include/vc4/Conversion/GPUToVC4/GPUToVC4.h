#ifndef VC4_CONVERSION_GPUTOVC4_GPUTOVC4_H
#define VC4_CONVERSION_GPUTOVC4_GPUTOVC4_H

#include <memory>

namespace mlir {
class Pass;

namespace vc4 {

std::unique_ptr<Pass> createConvertGPUToVC4Pass();
void registerGPUToVC4Passes();

} // namespace vc4
} // namespace mlir

#endif // VC4_CONVERSION_GPUTOVC4_GPUTOVC4_H
