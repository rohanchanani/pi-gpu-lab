//===- ValueSurfacePasses.h - VC4 value surface pass declarations -*- C++ -*-===//

#ifndef VC4_TRANSFORMS_VALUESURFACE_VALUESURFACEPASSES_H
#define VC4_TRANSFORMS_VALUESURFACE_VALUESURFACEPASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace vc4 {

std::unique_ptr<Pass> createVerifyValueSurfacePass();

void registerValueSurfacePasses();

} // namespace vc4
} // namespace mlir

#endif // VC4_TRANSFORMS_VALUESURFACE_VALUESURFACEPASSES_H
