add_mlir_dialect_library(VC4Dialect
  VC4Dialect.cpp
  VC4Enums.cpp
  VC4Types.cpp
  VC4Ops.cpp

  DEPENDS
    VC4AttrDefsIncGen
    VC4DialectIncGen
    VC4EnumsIncGen
    VC4TypesIncGen
    VC4OpsIncGen

  LINK_LIBS PUBLIC
    ${VC4_MLIR_LINK_LIBS}
)
