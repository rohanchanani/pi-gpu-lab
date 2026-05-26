# tmu_read_nop_write candidate

This candidate side is enabled. The harness builds the generated `input.mlir`
bundle, launches the TMU0 direct-memory read plus VPM/VDW writeback kernel on
hardware, and checks every copied word plus an output guard region against a
strict host oracle.
