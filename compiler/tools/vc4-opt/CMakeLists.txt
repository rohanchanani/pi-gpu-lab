add_mlir_tool(vc4-opt
  vc4-opt.cpp
  ../../lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp
)

add_dependencies(vc4-opt
  VC4OpsIncGen
  SSAVC4DialectIncGen
  SSAVC4EnumsIncGen
  SSAVC4AttrDefsIncGen
  SSAVC4TypesIncGen
  SSAVC4OpsIncGen
)

target_link_libraries(vc4-opt
  PRIVATE
    VC4Dialect
    SSAVC4Dialect
    MLIR
)

# The M3 slice-3 typed verifier invokes filtered lit roots directly under
# compiler/build/test.  Wire those build-tree roots from this CMake file, which
# is already reconfigured when the vc4-opt target is built, while leaving the
# scheduled VC4 sink and generic test drivers untouched.
function(vc4_configure_ssavc4_lit_root source_dir binary_dir)
  file(MAKE_DIRECTORY "${binary_dir}")
  file(GLOB vc4_ssavc4_lit_tests CONFIGURE_DEPENDS
    "${source_dir}/*.mlir")
  foreach(test_file ${vc4_ssavc4_lit_tests})
    get_filename_component(test_name "${test_file}" NAME)
    configure_file("${test_file}" "${binary_dir}/${test_name}" COPYONLY)
  endforeach()
  configure_file(
    "${source_dir}/lit.site.cfg.py.in"
    "${binary_dir}/lit.site.cfg.py"
    @ONLY)
endfunction()

vc4_configure_ssavc4_lit_root(
  "${CMAKE_CURRENT_SOURCE_DIR}/../../test/Conversion/SSAVC4ToVC4"
  "${CMAKE_BINARY_DIR}/test/Conversion/SSAVC4ToVC4")

vc4_configure_ssavc4_lit_root(
  "${CMAKE_CURRENT_SOURCE_DIR}/../../test/CodeGen/SSAVC4/Emit"
  "${CMAKE_BINARY_DIR}/test/CodeGen/SSAVC4/Emit")
