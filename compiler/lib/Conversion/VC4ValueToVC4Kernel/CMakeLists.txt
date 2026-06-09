add_mlir_library(VC4ValueToVC4Kernel
  VC4ValueToVC4Kernel.cpp

  DEPENDS
    VC4ValueDialectIncGen
    VC4ValueAttrDefsIncGen
    VC4ValueOpsIncGen
    VC4KernelDialectIncGen
    VC4KernelEnumsIncGen
    VC4KernelAttrDefsIncGen
    VC4KernelTypesIncGen
    VC4KernelOpsIncGen

  LINK_LIBS PUBLIC
    VC4ValueDialect
    VC4KernelDialect
    ${VC4_MLIR_LINK_LIBS}
)
