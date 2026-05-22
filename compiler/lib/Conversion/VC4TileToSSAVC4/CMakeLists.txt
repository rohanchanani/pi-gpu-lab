add_mlir_library(VC4TileToSSAVC4
  VC4TileToSSAVC4.cpp

  DEPENDS
    VC4TileDialectIncGen
    VC4TileEnumsIncGen
    VC4TileAttrDefsIncGen
    VC4TileTypesIncGen
    VC4TileOpsIncGen
    SSAVC4DialectIncGen
    SSAVC4EnumsIncGen
    SSAVC4AttrDefsIncGen
    SSAVC4TypesIncGen
    SSAVC4OpsIncGen

  LINK_LIBS PUBLIC
    VC4Dialect
    VC4TileDialect
    SSAVC4Dialect
    MLIR
)
