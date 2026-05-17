add_mlir_library(SSAVC4ToVC4
  SSAVC4ToVC4.cpp

  DEPENDS
    VC4OpsIncGen
    SSAVC4OpsIncGen

  LINK_LIBS PUBLIC
    VC4Dialect
    SSAVC4Dialect
    MLIR
)
