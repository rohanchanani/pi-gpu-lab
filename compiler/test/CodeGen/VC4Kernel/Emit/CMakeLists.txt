file(GLOB VC4KERNEL_CODEGEN_EMIT_TESTS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.mlir")
foreach(test_file ${VC4KERNEL_CODEGEN_EMIT_TESTS})
  get_filename_component(test_name "${test_file}" NAME)
  configure_file("${test_file}" "${CMAKE_CURRENT_BINARY_DIR}/${test_name}" COPYONLY)
endforeach()

set(VC4KERNEL_VECTOR_STORE_FIXTURE
  "${CMAKE_CURRENT_SOURCE_DIR}/../Hardware/Run/vector_store_smoke_vc4kernel/input.mlir")
set(VC4KERNEL_VECTOR_STORE_FIXTURE_OUT
  "${CMAKE_CURRENT_BINARY_DIR}/../Hardware/Run/vector_store_smoke_vc4kernel/input.mlir")
get_filename_component(VC4KERNEL_VECTOR_STORE_FIXTURE_OUT_DIR
  "${VC4KERNEL_VECTOR_STORE_FIXTURE_OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${VC4KERNEL_VECTOR_STORE_FIXTURE_OUT_DIR}")
configure_file("${VC4KERNEL_VECTOR_STORE_FIXTURE}"
  "${VC4KERNEL_VECTOR_STORE_FIXTURE_OUT}" COPYONLY)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  @ONLY)
