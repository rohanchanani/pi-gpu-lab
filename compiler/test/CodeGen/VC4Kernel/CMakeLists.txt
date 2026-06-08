file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py"
  COPYONLY)

file(GLOB VC4KERNEL_CODEGEN_ROOT_TESTS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.test")
foreach(test_file ${VC4KERNEL_CODEGEN_ROOT_TESTS})
  get_filename_component(test_name "${test_file}" NAME)
  configure_file(
    "${test_file}"
    "${CMAKE_CURRENT_BINARY_DIR}/${test_name}"
    COPYONLY)
endforeach()

file(GLOB VC4KERNEL_MIXED_ACCEPTANCE_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/Hardware/MixedAcceptance/*")
foreach(mixed_file ${VC4KERNEL_MIXED_ACCEPTANCE_FILES})
  get_filename_component(mixed_name "${mixed_file}" NAME)
  set(mixed_out
    "${CMAKE_CURRENT_BINARY_DIR}/Hardware/MixedAcceptance/${mixed_name}")
  get_filename_component(mixed_out_dir "${mixed_out}" DIRECTORY)
  file(MAKE_DIRECTORY "${mixed_out_dir}")
  configure_file("${mixed_file}" "${mixed_out}" COPYONLY)
endforeach()

file(GLOB VC4KERNEL_CODEGEN_SUPPORT_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/Support/*")
foreach(support_file ${VC4KERNEL_CODEGEN_SUPPORT_FILES})
  get_filename_component(support_name "${support_file}" NAME)
  set(support_out
    "${CMAKE_CURRENT_BINARY_DIR}/Support/${support_name}")
  get_filename_component(support_out_dir "${support_out}" DIRECTORY)
  file(MAKE_DIRECTORY "${support_out_dir}")
  configure_file("${support_file}" "${support_out}" COPYONLY)
endforeach()

add_subdirectory(Emit)
