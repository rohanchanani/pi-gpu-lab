file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  @ONLY)

file(GLOB VC4_VALUE_SURFACE_TESTS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.mlir"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.test")
foreach(test_file ${VC4_VALUE_SURFACE_TESTS})
  get_filename_component(test_name "${test_file}" NAME)
  configure_file(
    "${test_file}"
    "${CMAKE_CURRENT_BINARY_DIR}/${test_name}"
    COPYONLY)
endforeach()
