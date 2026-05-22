add_mlir_dialect_library(VC4TileDialect
  VC4TileDialect.cpp
  VC4TileAttrs.cpp
  VC4TileOps.cpp
  VC4TileTypes.cpp

  DEPENDS
    VC4TileDialectIncGen
    VC4TileEnumsIncGen
    VC4TileAttrDefsIncGen
    VC4TileTypesIncGen
    VC4TileOpsIncGen

  LINK_LIBS PUBLIC
    VC4Dialect
    MLIR
)
