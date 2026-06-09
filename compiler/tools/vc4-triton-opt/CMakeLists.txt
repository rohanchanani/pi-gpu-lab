# Optional Triton C++ frontend driver.  This directory should be added only when
# VC4_ENABLE_TRITON_CPP_FRONTEND=ON and the caller has provided the Triton source
# and build include directories plus the transitive Triton C++ link closure
# proven by the Phase 7.5 toolchain-lane parse proof.

add_mlir_tool(vc4-triton-opt
  vc4-triton-opt.cpp
)

add_dependencies(vc4-triton-opt
  VC4ValueDialectIncGen
  VC4ValueAttrDefsIncGen
  VC4ValueOpsIncGen
)

target_link_libraries(vc4-triton-opt
  PRIVATE
    TritonToVC4Value
    VC4ValueDialect
    VC4TritonFrontendDeps
    ${VC4_MLIR_TOOL_LINK_LIBS}
)
