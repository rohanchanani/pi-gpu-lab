#ifndef VC4TILE_BASE_TD
#define VC4TILE_BASE_TD

include "mlir/IR/OpBase.td"

// VC4Tile is the M4 tile/kernel layer above SSAVC4.  This scaffold defines
// only the dialect identity; operation contracts and lowering arrive in later
// M4 slices.
def VC4Tile_Dialect : Dialect {
  let name = "vc4tile";
  let cppNamespace = "::mlir::vc4tile";
  let summary = "VideoCore IV tile kernel dialect";
  let description = [{
    VC4Tile represents target-specific 16-lane tile/kernel intent above the
    SSAVC4 machine-SSA layer.  It is separate from both the scheduled `vc4`
    artifact sink and future producer-adapter milestones.
  }];
}

#endif // VC4TILE_BASE_TD
