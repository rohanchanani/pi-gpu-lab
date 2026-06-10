file(GLOB VC4VALUE_CODEGEN_EMIT_TESTS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.mlir"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.test")
foreach(test_file ${VC4VALUE_CODEGEN_EMIT_TESTS})
  get_filename_component(test_name "${test_file}" NAME)
  configure_file("${test_file}" "${CMAKE_CURRENT_BINARY_DIR}/${test_name}" COPYONLY)
endforeach()

file(GLOB VC4VALUE_HARDWARE_INPUTS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/../Hardware/Run/*/input.mlir")
foreach(input_file ${VC4VALUE_HARDWARE_INPUTS})
  get_filename_component(fixture_dir "${input_file}" DIRECTORY)
  get_filename_component(fixture_name "${fixture_dir}" NAME)
  set(input_out
    "${CMAKE_CURRENT_BINARY_DIR}/../Hardware/Run/${fixture_name}/input.mlir")
  get_filename_component(input_out_dir "${input_out}" DIRECTORY)
  file(MAKE_DIRECTORY "${input_out_dir}")
  configure_file("${input_file}" "${input_out}" COPYONLY)
endforeach()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  @ONLY)
