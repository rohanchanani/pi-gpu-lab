# Keep the SSAVC4 dialect lit root materialized in the build tree.  The typed
# verifier runs llvm-lit against compiler/build/test/Dialect/SSAVC4 directly,
# so copy every active source-tree .mlir test into this binary directory.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  @ONLY)

file(GLOB SSAVC4_DIALECT_MLIR_TESTS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.mlir")
foreach(test_file ${SSAVC4_DIALECT_MLIR_TESTS})
  get_filename_component(test_name "${test_file}" NAME)
  configure_file(
    "${test_file}"
    "${CMAKE_CURRENT_BINARY_DIR}/${test_name}"
    COPYONLY)
endforeach()
