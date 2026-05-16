# Keep the scaffold lit root materialized in the build tree.  The typed
# verifier runs llvm-lit against compiler/build/test/Dialect/SSAVC4 directly,
# so this directory and its lit config must exist after CMake reconfiguration.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  @ONLY)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/show-dialects.mlir"
  "${CMAKE_CURRENT_BINARY_DIR}/show-dialects.mlir"
  COPYONLY)
