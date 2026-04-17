add_mlir_dialect_library(VC4
  VC4Dialect.cpp
  VC4Ops.cpp

  DEPENDS
  MLIRVC4OpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
