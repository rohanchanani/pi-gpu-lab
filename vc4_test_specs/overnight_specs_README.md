# VC4 overnight test specifications

This bundle contains 20 hardware-run test specifications for the VC4 bundle autorunner.

The manifest order starts with hardware/runtime primitives, then shared-memory and image/scientific kernels, then ML/attention kernels, and finally a divergent masked-loop Mandelbrot stress test.

These files are specifications only. The autorunner and god prompt generate the actual hardware-run bundles.
