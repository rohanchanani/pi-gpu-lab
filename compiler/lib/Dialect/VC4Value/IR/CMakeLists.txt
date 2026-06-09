add_mlir_dialect_library(VC4ValueDialect
  VC4ValueDialect.cpp
  VC4ValueOps.cpp

  DEPENDS
    VC4ValueDialectIncGen
    VC4ValueAttrDefsIncGen
    VC4ValueOpsIncGen

  LINK_LIBS PUBLIC
    ${VC4_MLIR_LINK_LIBS}
)
