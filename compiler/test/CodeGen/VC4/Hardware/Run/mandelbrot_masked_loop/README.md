
mandelbrot_masked_loop

Validates divergent SIMD-style masked loop behavior by computing Mandelbrot iteration counts for a small image.

For each pixel (x, y), the kernel maps to a complex coordinate:

cx = min_x + x * step_x

cy = min_y + y * step_y

It then iterates:

z = z^2 + c

continue while |z|^2 <= 4 and iter < max_iter

The output is the first escape iteration count, or max_iter for points that do not escape. The intended hardware implementation represents divergent per-lane control flow using predicates/masks: escaped lanes stop updating their z values and keep their first escape count while still participating in the vector loop.

This reference bundle uses deterministic windows chosen away from especially brittle boundary-sensitive views. The stable oracle checks exact integer counts, guard preservation, one runtime allocation, and repeated launches.
