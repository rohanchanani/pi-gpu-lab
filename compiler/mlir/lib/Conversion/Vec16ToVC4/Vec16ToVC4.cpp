#include "pi_gpu/Conversion/Vec16ToVC4/Vec16ToVC4.h"
#include "pi_gpu/VC4/VC4Dialect.h"

namespace pi_gpu {

namespace {

class ConvertVec16ToVC4Pass;

class ConvertVec16ToVC4Pass : public mlir::Pass {
 public:
  static constexpr const char *getArgument() { return "convert-vec16-to-vc4"; }
  static constexpr const char *getDescription() {
    return "Lower width-16 vector groups to a tiny staged-memory vc4 dialect";
  }

  void runOnOperation() {
    // Seed only: later work will rewrite vec16 transfer/arithmetic structure
    // into vc4 uniform, DMA/VPM staging, and simple backend arithmetic ops.
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createConvertVec16ToVC4Pass() {
  return std::make_unique<ConvertVec16ToVC4Pass>();
}

void registerConvertVec16ToVC4Pass() {
  // Placeholder until pass registration is wired to MLIR's registry helpers.
}

}  // namespace pi_gpu
