add_mlir_library(TritonToVC4Value
  TritonToVC4Value.cpp

  DEPENDS
    VC4ValueDialectIncGen
    VC4ValueAttrDefsIncGen
    VC4ValueOpsIncGen

  LINK_LIBS PUBLIC
    VC4ValueDialect
    VC4TritonFrontendDeps
    ${VC4_MLIR_LINK_LIBS}
)
