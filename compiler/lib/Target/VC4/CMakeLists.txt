add_mlir_library(VC4ArtifactEmitter
  VC4ArtifactEmitter.cpp

  DEPENDS
    VC4OpsIncGen

  LINK_LIBS PUBLIC
    VC4Dialect
    MLIR
)

# Bootstrap this slice through the already-reached compiler/lib tree.  The
# standalone tool's own CMake file remains under compiler/tools/vc4-codegen/ so
# later slices can move to a conventional compiler/tools aggregate without
# changing the public target name.
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/vc4-codegen
  ${CMAKE_CURRENT_BINARY_DIR}/vc4-codegen-tool
)
