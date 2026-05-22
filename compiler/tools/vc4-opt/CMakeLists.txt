add_mlir_tool(vc4-opt
  vc4-opt.cpp
  ${PROJECT_SOURCE_DIR}/lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp
  ${PROJECT_SOURCE_DIR}/lib/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.cpp
)

add_dependencies(vc4-opt
  VC4OpsIncGen
  SSAVC4DialectIncGen
  SSAVC4EnumsIncGen
  SSAVC4AttrDefsIncGen
  SSAVC4TypesIncGen
  SSAVC4OpsIncGen
  VC4TileDialectIncGen
  VC4TileEnumsIncGen
  VC4TileAttrDefsIncGen
  VC4TileTypesIncGen
  VC4TileOpsIncGen
)

target_link_libraries(vc4-opt
  PRIVATE
    VC4Dialect
    SSAVC4Dialect
    VC4TileDialect
    MLIR
)
