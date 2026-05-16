#ifndef SSAVC4_OPS_TD
#define SSAVC4_OPS_TD

include "vc4/Dialect/SSAVC4/IR/SSAVC4Dialect.td"
include "mlir/IR/OpBase.td"

// Intentionally empty for the scaffold slice.  Later M3 slices add the
// operation surface vertically with verifiers, effects, lowering, and tests.
class SSAVC4_Op<string mnemonic, list<Trait> traits = []>
    : Op<SSAVC4_Dialect, mnemonic, traits>;

#endif // SSAVC4_OPS_TD
