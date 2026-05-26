# memory_output candidate

This candidate side is enabled. The harness builds the generated `input.mlir`
bundle, launches the VPM/VDW output kernel on hardware, and checks every output
word plus an output guard region against a strict host oracle.
