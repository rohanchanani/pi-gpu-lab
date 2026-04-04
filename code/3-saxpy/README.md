# 3-saxpy

Run `bash run.sh` from this directory to:

1. assemble `saxpy.qasm` with `vc4asm`
2. build and run the bare-metal GPU test via `make`

The test harness:
- runs GPU SAXPY (`y = a*x + y`)
- runs a CPU reference implementation over the same inputs
- compares the outputs with a small float epsilon
- prints CPU and GPU timings plus the measured speedup
