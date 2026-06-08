add_mlir_dialect_library(VC4ValueDialect
  VC4ValueDialect.cpp

  DEPENDS
    VC4ValueDialectIncGen

  LINK_LIBS PUBLIC
    MLIR
)
