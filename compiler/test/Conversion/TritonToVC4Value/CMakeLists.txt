file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py"
  COPYONLY)

file(GLOB TRITON_TO_VC4VALUE_TESTS LIST_DIRECTORIES false CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.test"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.json")
foreach(test_file ${TRITON_TO_VC4VALUE_TESTS})
  get_filename_component(test_name "${test_file}" NAME)
  configure_file(
    "${test_file}"
    "${CMAKE_CURRENT_BINARY_DIR}/${test_name}"
    COPYONLY)
endforeach()

file(GLOB TRITON_TO_VC4VALUE_ROBUSTNESS_TESTS LIST_DIRECTORIES false CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/Robustness/*.mlir")
foreach(test_file ${TRITON_TO_VC4VALUE_ROBUSTNESS_TESTS})
  get_filename_component(test_name "${test_file}" NAME)
  set(test_out "${CMAKE_CURRENT_BINARY_DIR}/Robustness/${test_name}")
  get_filename_component(test_out_dir "${test_out}" DIRECTORY)
  file(MAKE_DIRECTORY "${test_out_dir}")
  configure_file("${test_file}" "${test_out}" COPYONLY)
endforeach()

file(GLOB TRITON_TO_VC4VALUE_REDUCTION_TESTS LIST_DIRECTORIES false CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/Reductions/*.mlir")
foreach(test_file ${TRITON_TO_VC4VALUE_REDUCTION_TESTS})
  get_filename_component(test_name "${test_file}" NAME)
  set(test_out "${CMAKE_CURRENT_BINARY_DIR}/Reductions/${test_name}")
  get_filename_component(test_out_dir "${test_out}" DIRECTORY)
  file(MAKE_DIRECTORY "${test_out_dir}")
  configure_file("${test_file}" "${test_out}" COPYONLY)
endforeach()

file(GLOB TRITON_TO_VC4VALUE_SUPPORT_FILES LIST_DIRECTORIES false CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/Support/*")
foreach(support_file ${TRITON_TO_VC4VALUE_SUPPORT_FILES})
  get_filename_component(support_name "${support_file}" NAME)
  set(support_out
    "${CMAKE_CURRENT_BINARY_DIR}/Support/${support_name}")
  get_filename_component(support_out_dir "${support_out}" DIRECTORY)
  file(MAKE_DIRECTORY "${support_out_dir}")
  configure_file("${support_file}" "${support_out}" COPYONLY)
endforeach()
