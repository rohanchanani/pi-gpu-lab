add_mlir_dialect_library(SSAVC4Dialect
  SSAVC4Dialect.cpp
  SSAVC4Attrs.cpp
  SSAVC4Enums.cpp
  SSAVC4Types.cpp
  SSAVC4Ops.cpp

  DEPENDS
    SSAVC4DialectIncGen
    SSAVC4EnumsIncGen
    SSAVC4AttrDefsIncGen
    SSAVC4TypesIncGen
    SSAVC4OpsIncGen

  LINK_LIBS PUBLIC
    VC4Dialect
    ${VC4_MLIR_LINK_LIBS}
)
