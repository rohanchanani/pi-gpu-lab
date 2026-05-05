add_mlir_library(VC4Target
 EmitBundle.cpp

 LINK_LIBS PUBLIC
 VC4Dialect
 MLIR
)
