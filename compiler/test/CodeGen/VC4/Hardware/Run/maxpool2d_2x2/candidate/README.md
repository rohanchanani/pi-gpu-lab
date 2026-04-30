
maxpool2d_2x2 candidate

A compiler-generated candidate for this test should implement 2x2 max pooling over one row-major float image per launch.

The intended hardware shape uses 12 QPUs and 16 lanes. Each lane maps a flat output index to (oy, ox), computes up to four input indices, substitutes a very negative value for missing odd-edge elements, computes the maximum, and stores coalesced output through the conservative VPM/VDW store path.

The public launch API exposes only semantic image buffers and dimensions. It must not expose QPU ids, raw uniforms, VPM rows, VDW setup, or scheduler details.
