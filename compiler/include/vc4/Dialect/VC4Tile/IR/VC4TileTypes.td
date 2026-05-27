#ifndef VC4TILE_TYPES_TD
#define VC4TILE_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "vc4/Dialect/VC4Tile/IR/VC4TileBase.td"

class VC4Tile_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<VC4Tile_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def VC4Tile_SharedTileType : VC4Tile_Type<"SharedTile", "shared_tile"> {
  let summary = "opaque shared VPM tile handle";
  let description = [{
    Opaque VC4Tile core handle for a statically allocated shared-VPM tile.  It
    preserves the M4 shared VPM contract and remains distinct from ergonomic M5
    tile descriptors.
  }];
  let assemblyFormat = "";
}

def VC4Tile_TileType : VC4Tile_Type<"Tile", "tile"> {
  let summary = "M5 ergonomic metadata tile handle";
  let description = [{
    Metadata-only ergonomic tile handle used by M5 tile descriptors.  Logical
    shape, rank, memory space, layout, role, storage type, accumulator type,
    precision policy and physical-storage marker are carried by verified operation attributes
    in M5 so unsupported sub-32 forms can be diagnosed before executable copy
    or compute lowering exists.
  }];
  let assemblyFormat = "";
}

#endif // VC4TILE_TYPES_TD
