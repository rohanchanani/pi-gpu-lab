add_mlir_tool(vc4-opt
  vc4-opt.cpp
)

add_dependencies(vc4-opt
  SSAVC4DialectIncGen
  SSAVC4OpsIncGen
)

target_link_libraries(vc4-opt
  PRIVATE
    VC4Dialect
    SSAVC4Dialect
    MLIR
)
