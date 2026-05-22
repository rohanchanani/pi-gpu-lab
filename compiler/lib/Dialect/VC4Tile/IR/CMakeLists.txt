add_mlir_dialect_library(VC4TileDialect
  VC4TileDialect.cpp

  DEPENDS
    VC4TileDialectIncGen

  LINK_LIBS PUBLIC
    MLIR
)
