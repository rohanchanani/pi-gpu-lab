add_mlir_translation_library(VC4Asm
  Artifacts.cpp
  KernelModel.cpp
  Launcher.cpp
  QASM.cpp
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

target_compile_definitions(VC4Asm PRIVATE VC4_SOURCE_DIR="${VC4_SOURCE_DIR}")
target_compile_definitions(obj.VC4Asm PRIVATE VC4_SOURCE_DIR="${VC4_SOURCE_DIR}")
