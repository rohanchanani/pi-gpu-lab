add_mlir_conversion_library(GPUToVC4
  GPUToVC4.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
