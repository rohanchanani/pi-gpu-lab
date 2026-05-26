# saxpy_16 candidate

This candidate side is enabled. The harness builds the generated `input.mlir`
bundle, launches the multi-wave TMU/f32 ALU/VDW SAXPY kernel on hardware, and
compares all 768 output values against a strict scalar host `alpha * x + y`
oracle while also checking an output guard region.
