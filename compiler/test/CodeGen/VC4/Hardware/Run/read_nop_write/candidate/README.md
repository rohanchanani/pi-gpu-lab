# read_nop_write candidate

This candidate side is enabled. The harness builds the generated `input.mlir`
bundle, launches the VC4 VDR/VPM/VDW read-copy-write kernel on hardware, and
checks every copied word plus an output guard region against a strict host
oracle.
