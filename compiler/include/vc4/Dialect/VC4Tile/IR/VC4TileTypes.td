#ifndef VC4TILE_TYPES_TD
#define VC4TILE_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "vc4/Dialect/VC4Tile/IR/VC4TileBase.td"

class VC4Tile_Type<string name, string typeMnemonic>
    : TypeDef<VC4Tile_Dialect, name> {
  let mnemonic = typeMnemonic;
}

// Opaque handle for one row-granular shared VPM tile allocation.  Concrete VPM
// row assignment and setup words are owned by later VC4Tile-to-SSAVC4 lowering.
def VC4Tile_SharedTileType : VC4Tile_Type<"SharedTile", "shared_tile"> {
  let summary = "opaque VC4Tile shared VPM tile handle";
}

#endif // VC4TILE_TYPES_TD
