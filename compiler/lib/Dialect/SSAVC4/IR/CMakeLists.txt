add_mlir_dialect_library(SSAVC4Dialect
  SSAVC4Dialect.cpp
  SSAVC4Ops.cpp

  DEPENDS
    SSAVC4DialectIncGen
    SSAVC4OpsIncGen

  LINK_LIBS PUBLIC
    MLIR
)
