configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
    "VC4_SOURCE_DIR"
    "VC4_BINARY_DIR"
    "VC4_TOOLS_DIR"
    "LLVM_TOOLS_DIR"
    "MLIR_TOOLS_DIR"
)

add_lit_testsuite(check-vc4 "Running the VC4 regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
    vc4-opt
)
