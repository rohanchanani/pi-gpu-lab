file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Run")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../Support")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.local.cfg.py"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.local.cfg.py"
  COPYONLY)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Run/lit.local.cfg.py"
  "${CMAKE_CURRENT_BINARY_DIR}/Run/lit.local.cfg.py"
  COPYONLY)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/../Support/run_vc4tile_candidate_codegen_test.sh"
  "${CMAKE_CURRENT_BINARY_DIR}/../Support/run_vc4tile_candidate_codegen_test.sh"
  COPYONLY)

file(GLOB_RECURSE VC4TILE_HARDWARE_TESTS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/Run/*")
foreach(test_file ${VC4TILE_HARDWARE_TESTS})
  if(NOT IS_DIRECTORY "${test_file}")
    file(RELATIVE_PATH rel "${CMAKE_CURRENT_SOURCE_DIR}" "${test_file}")
    configure_file(
      "${test_file}"
      "${CMAKE_CURRENT_BINARY_DIR}/${rel}"
      COPYONLY)
  endif()
endforeach()
