# sfu_recip candidate

This candidate side is enabled. The harness builds the generated `input.mlir`
bundle, launches the VC4 SFU reciprocal kernel on hardware, and checks every
lane against the host `1.0f / x` oracle with a tight bound for the SFU
approximation.
