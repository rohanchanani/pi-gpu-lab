add_mlir_tool(vc4-opt
  vc4-opt.cpp
)

target_link_libraries(vc4-opt
  PRIVATE
    VC4Dialect
    MLIR
)
