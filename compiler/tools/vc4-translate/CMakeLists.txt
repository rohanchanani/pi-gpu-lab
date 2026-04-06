add_mlir_tool(vc4-translate
  vc4-translate.cpp
)

target_link_libraries(vc4-translate PRIVATE
  MLIRRegisterAllDialects
  MLIRTranslateLib
  VC4
  VC4Asm
)

mlir_check_all_link_libraries(vc4-translate)
