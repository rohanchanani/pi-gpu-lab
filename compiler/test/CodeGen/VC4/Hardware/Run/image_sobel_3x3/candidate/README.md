
image_sobel_3x3 candidate

A compiler-generated candidate for this test should implement the same semantic Sobel transform as the reference:

gx = -p00 + p02 - 2*p10 + 2*p12 - p20 + p22

gy = -p00 - 2*p01 - p02 + p20 + 2*p21 + p22

out = min(255, abs(gx) + abs(gy))

Inputs are row-major uint32_t grayscale pixels using the low 8 bits. Boundary pixels use clamp-to-edge addressing. Outputs are exact uint32_t values.
