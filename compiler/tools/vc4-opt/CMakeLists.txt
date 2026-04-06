add_mlir_tool(vc4-opt
  vc4-opt.cpp
)

target_link_libraries(vc4-opt PRIVATE
  GPUToVC4
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIROptLib
  VC4
)

mlir_check_all_link_libraries(vc4-opt)
