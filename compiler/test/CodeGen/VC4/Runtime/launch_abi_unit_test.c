#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef uint32_t vc4_deviceptr_t;
typedef struct vc4_dim3 { uint32_t x, y, z; } vc4_dim3;

static uint32_t pack_f32(float f) {
    uint32_t bits = 0;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

int main(void) {
    uint32_t unif[6] = {0};
    vc4_deviceptr_t x = 0x100u;
    vc4_deviceptr_t y = 0x200u;
    float alpha = 2.5f;
    uint32_t n = 17u;
    unif[0] = x;
    unif[1] = y;
    unif[2] = pack_f32(alpha);
    unif[3] = n;
    unif[4] = 3u;  /* qpu_id suffix */
    unif[5] = 12u; /* num_qpus suffix */
    int ok = unif[0] == 0x100u && unif[1] == 0x200u && unif[2] == 0x40200000u && unif[3] == 17u && unif[4] == 3u && unif[5] == 12u;
    printf("LAUNCH_ABI_UNIT_RESULT status=%s uniforms=%u,%u,%08x,%u,%u,%u\n",
           ok ? "PASS" : "FAIL", unif[0], unif[1], unif[2], unif[3], unif[4], unif[5]);
    return ok ? 0 : 1;
}
