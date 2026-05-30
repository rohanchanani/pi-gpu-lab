//===- VC4KernelToSSAVC4.h - VC4Kernel to SSAVC4 passes -------*- C++ -*-===//

#ifndef VC4_CONVERSION_VC4KERNELTOSSAVC4_VC4KERNELTOSSAVC4_H
#define VC4_CONVERSION_VC4KERNELTOSSAVC4_VC4KERNELTOSSAVC4_H

#include <memory>

namespace mlir {
class Pass;

namespace vc4 {

std::unique_ptr<Pass> createVerifyVC4KernelPass();
std::unique_ptr<Pass> createConvertVC4KernelToSSAVC4Pass();

void registerConvertVC4KernelToSSAVC4Pass();

} // namespace vc4
} // namespace mlir

#endif // VC4_CONVERSION_VC4KERNELTOSSAVC4_VC4KERNELTOSSAVC4_H
