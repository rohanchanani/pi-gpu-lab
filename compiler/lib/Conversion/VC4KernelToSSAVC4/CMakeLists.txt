add_mlir_library(VC4KernelToSSAVC4
  VC4KernelToSSAVC4.cpp

  DEPENDS
    VC4KernelDialectIncGen
    VC4KernelEnumsIncGen
    VC4KernelAttrDefsIncGen
    VC4KernelTypesIncGen
    VC4KernelOpsIncGen
    SSAVC4OpsIncGen

  LINK_LIBS PUBLIC
    VC4Dialect
    SSAVC4Dialect
    VC4KernelDialect
    ${VC4_MLIR_LINK_LIBS}
)
