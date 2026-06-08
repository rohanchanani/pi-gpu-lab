file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py"
  COPYONLY)

file(GLOB VC4VALUE_CODEGEN_SUPPORT_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/Support/*")
foreach(support_file ${VC4VALUE_CODEGEN_SUPPORT_FILES})
  get_filename_component(support_name "${support_file}" NAME)
  set(support_out
    "${CMAKE_CURRENT_BINARY_DIR}/Support/${support_name}")
  get_filename_component(support_out_dir "${support_out}" DIRECTORY)
  file(MAKE_DIRECTORY "${support_out_dir}")
  configure_file("${support_file}" "${support_out}" COPYONLY)
endforeach()

add_subdirectory(Emit)
