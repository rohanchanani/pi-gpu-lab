add_mlir_library(VC4ValueSurfaceTransforms
  ValueSurfaceVerification.cpp

  LINK_LIBS PUBLIC
    VC4ValueDialect
    MLIR
)
