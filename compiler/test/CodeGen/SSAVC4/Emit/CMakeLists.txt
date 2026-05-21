file(GLOB SSAVC4_CODEGEN_EMIT_TESTS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.mlir")
foreach(test_file ${SSAVC4_CODEGEN_EMIT_TESTS})
  get_filename_component(test_name "${test_file}" NAME)
  configure_file("${test_file}" "${CMAKE_CURRENT_BINARY_DIR}/${test_name}" COPYONLY)
endforeach()
file(GLOB SSAVC4_CODEGEN_EMIT_INPUTS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/Inputs/*")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Inputs")
foreach(input_file ${SSAVC4_CODEGEN_EMIT_INPUTS})
  get_filename_component(input_name "${input_file}" NAME)
  configure_file("${input_file}" "${CMAKE_CURRENT_BINARY_DIR}/Inputs/${input_name}" COPYONLY)
endforeach()
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  @ONLY)
