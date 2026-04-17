#ifndef VC4_CONVERSION_GPUTOVC4_GPUTOVC4_H
#define VC4_CONVERSION_GPUTOVC4_GPUTOVC4_H

#include <memory>

namespace mlir {
class Pass;

namespace vc4 {

std::unique_ptr<Pass> createLowerGPUToVC4Pass();

// Deprecated compatibility entry point. New code should use
// createLowerGPUToVC4Pass().
[[deprecated("use createLowerGPUToVC4Pass() instead")]]
std::unique_ptr<Pass> createLowerGPUSaxpyToVC4Pass();

void registerGPUToVC4Passes();

} // namespace vc4
} // namespace mlir

#endif // VC4_CONVERSION_GPUTOVC4_GPUTOVC4_H
