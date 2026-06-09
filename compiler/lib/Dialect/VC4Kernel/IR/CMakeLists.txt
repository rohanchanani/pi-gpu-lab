add_mlir_dialect_library(VC4KernelDialect
  VC4KernelDialect.cpp
  VC4KernelAttrs.cpp
  VC4KernelOps.cpp
  VC4KernelTypes.cpp

  DEPENDS
    VC4KernelDialectIncGen
    VC4KernelEnumsIncGen
    VC4KernelAttrDefsIncGen
    VC4KernelTypesIncGen
    VC4KernelOpsIncGen

  LINK_LIBS PUBLIC
    ${VC4_MLIR_LINK_LIBS}
)
