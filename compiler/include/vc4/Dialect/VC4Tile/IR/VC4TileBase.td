#ifndef VC4TILE_BASE_TD
#define VC4TILE_BASE_TD

include "mlir/IR/OpBase.td"

// VC4Tile is the M4 tile/kernel layer above SSAVC4.  It exposes the
// target-specific 16-lane tile contract, but deliberately hides TMU/VDW setup
// words, raw semaphore protocols, physical registers, scheduling hazards,
// branch delay slots, and QASM.
def VC4Tile_Dialect : Dialect {
  let name = "vc4tile";
  let cppNamespace = "::mlir::vc4tile";
  let summary = "VideoCore IV tile kernel dialect";
  let description = [{
    VC4Tile represents target-specific 16-lane tile/kernel intent above the
    SSAVC4 machine-SSA layer.  It is separate from both the scheduled `vc4`
    artifact sink and future producer-adapter milestones.
  }];
  // Keep VC4Tile M4 operations on classic dictionary-backed inherent
  // attributes.  The Homebrew LLVM 22 ODS generator defaults to
  // properties for op attributes, which emits BytecodeOpInterface and
  // getProperties()-based accessors that are not part of this local
  // build surface.  M4 only needs stable textual parsing/printing and
  // deterministic verifiers, so explicit opt-out is the smallest
  // compatible contract.
  let usePropertiesForAttributes = 0;
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // VC4TILE_BASE_TD
