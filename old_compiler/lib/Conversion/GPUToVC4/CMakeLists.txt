add_mlir_conversion_library(GPUToVC4
  GPUToVC4.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRVectorDialect
  VC4
)
