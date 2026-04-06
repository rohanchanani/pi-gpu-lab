add_mlir_translation_library(VC4Asm
  VC4Asm.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  VC4
)
