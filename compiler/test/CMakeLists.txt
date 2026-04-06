set(VC4_TEST_DEPENDS
  FileCheck
  count
  not
  vc4-opt
  vc4-translate
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-vc4 "Running the VC4 regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${VC4_TEST_DEPENDS}
)

add_lit_testsuites(VC4 ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${VC4_TEST_DEPENDS}
)
