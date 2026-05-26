
warp_reduce_sum

warp_reduce_sum validates a generated VC4 16-lane floating-point reduction primitive.

This is the per-QPU lane-reduction building block used before block-wide reductions. Each logical request loads one 16-lane vector, masks inactive tail lanes to zero in the QPU program, reduces the active lanes to one sum, replicates that sum across the vector, and stores the valid prefix through VPM/VDW.

The semantic operation is:

for each logical vector v:
  sum = Σ active_lanes input[v * 16 + lane]
  out[v * 16 + lane] = sum for active lanes

The final partial vector masks inactive lanes to zero before reduction and stores only the logical active prefix through dynamic VDW DEPTH.

The hardware path exercised here is:

TMU0 direct memory lookup
-> QPU vector ALU reduction by rotate/add sequence
-> replicated sum vector
-> mutex-protected VPM staging
-> dynamic-depth VDW DMA store

This test does not validate cross-QPU reductions, VPM shared memory, barriers, arbitrary segmented reductions, or integer reductions. It is only the per-QPU lane-reduction primitive.

The candidate harness launches the generated VC4 kernel on hardware and compares every active output element against an exact host oracle. The input values are binary-exact eighths, so the host and QPU sums are required to match exactly; `max_abs_diff` must be `0.0`.

The result line also checks real generated-runtime counters: one program allocation, one code upload, thirteen launches, zero launch failures, matching checksums, and untouched output sentinels.
