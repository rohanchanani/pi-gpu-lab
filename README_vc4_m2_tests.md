# VC4 M2 test-suite bundle

This bundle installs the M2 test scaffolding described in the planning pass:
manifest-v2 lit tests, runtime heap/launch ABI unit tests, M2-ready candidate
support plumbing, CUDA-like candidate harnesses for saxpy_full and the new
multi-kernel fixtures, and two new hardware fixtures:

- multi_kernel_minimal: minimal_thrend + memory_output
- multi_kernel_chain: memory_output -> read_nop_write

Most new lit tests intentionally fail until the compiler implements the M2
backend/runtime contracts. The scripts and C unit tests can still be syntax- and
source-product-checked immediately.
