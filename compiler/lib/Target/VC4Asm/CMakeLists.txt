add_mlir_translation_library(VC4Asm
  VC4Asm.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRSCFDialect
  MLIRSupport
  MLIRTranslateLib
  MLIRVectorDialect
  VC4
)
