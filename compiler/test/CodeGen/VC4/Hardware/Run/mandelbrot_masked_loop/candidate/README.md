
mandelbrot_masked_loop candidate

A compiler-generated candidate for this test should implement a masked SIMD Mandelbrot loop.

The intended hardware shape uses 12 QPUs and 16 lanes. A vector chunk maps lanes to flat pixel indices. For each active lane:

Compute (x, y) from the flat pixel index.

Compute cx and cy.

Initialize zx = 0, zy = 0, iter = 0, and active = true.

Iterate while any lane remains active and the per-lane iteration count is less than max_iter.

Update zx and zy only for active lanes.

When a lane first has zx*zx + zy*zy > 4, record the current iteration count and deactivate that lane.

Store final counts through VPM/VDW with dynamic tail depth.

The public launch API exposes only semantic output geometry and view-window parameters. It must not expose physical QPU ids, raw uniforms, VPM rows, VDW setup, or scheduler registers.
