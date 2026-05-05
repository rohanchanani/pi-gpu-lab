add_mlir_tool(vc4-codegen
 vc4-codegen.cpp
)

target_link_libraries(vc4-codegen
 PRIVATE
 VC4Target
 VC4Dialect
 MLIR
)
