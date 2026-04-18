add_mlir_dialect_library(VC4Dialect
  VC4Dialect.cpp
  VC4Types.cpp

  DEPENDS
    VC4DialectIncGen

  LINK_LIBS PUBLIC
    MLIR
)
