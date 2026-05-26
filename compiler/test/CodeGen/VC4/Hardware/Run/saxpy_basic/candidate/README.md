# saxpy_basic candidate

This candidate side is enabled. The harness builds the generated `input.mlir`
bundle, launches the VDR/VPM/f32 ALU/VDW SAXPY kernel on hardware, and compares
all 192 output values against a strict scalar host `alpha * x + y` oracle while
also checking an output guard region.
