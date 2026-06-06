#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define BUFFER_N (LANES + 16u)
#define I_SENTINEL 0xdeadbeefu
#define F_SENTINEL_BITS 0x7fc0deadu

static const uint32_t i_bits[LANES] = {
    0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u,
    0x7fc00000u, 0xff800001u, 0x55555555u, 0xaaaaaaaau,
    0x00000001u, 0x7f800000u, 0xff800000u, 0x40490fdbu,
    0xc0490fdbu, 0x3f000000u, 0xbf000000u, 0x12345678u
};

static const uint32_t f_bits[LANES] = {
    0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u,
    0x7fc00000u, 0xff800001u, 0x55555555u, 0xaaaaaaaau,
    0x7f800000u, 0xff800000u, 0x00800000u, 0x80800000u,
    0x3f7fffffu, 0xbf7fffffu, 0x01020304u, 0xfedcba98u
};

static uint32_t out_i32[BUFFER_N];
static float f_values[LANES];
static float out_f32[BUFFER_N];

static float bits_to_float(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = bits;
    return v.f;
}

static uint32_t float_to_bits(float value) {
    union {
        uint32_t u;
        float f;
    } v;
    v.f = value;
    return v.u;
}

static void fill_buffers(void) {
    for (uint32_t lane = 0; lane < LANES; ++lane)
        f_values[lane] = bits_to_float(f_bits[lane]);
    for (uint32_t i = 0; i < BUFFER_N; ++i) {
        out_i32[i] = I_SENTINEL;
        out_f32[i] = bits_to_float(F_SENTINEL_BITS);
    }
}

static int verify_outputs(int *checksum_accum) {
    int mismatches = 0;
    *checksum_accum = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        uint32_t got_i = out_i32[lane];
        uint32_t got_f = float_to_bits(out_f32[lane]);
        uint32_t expected_i = i_bits[lane];
        uint32_t expected_f = f_bits[lane];
        *checksum_accum += (int)(got_i & 0xffffu);
        *checksum_accum += (int)(got_f & 0xffffu);
        if (got_i != expected_i || got_f != expected_f) {
            if (mismatches < 8) {
                printk("ERROR: fragment_bitcast lane=%d got_i=%x exp_i=%x got_f_bits=%x exp_f_bits=%x\n",
                       (int)lane, got_i, expected_i, got_f, expected_f);
            }
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = LANES; i < BUFFER_N; ++i) {
        uint32_t got_f = float_to_bits(out_f32[i]);
        if (out_i32[i] != I_SENTINEL || got_f != F_SENTINEL_BITS) {
            if (mismatches < 8) {
                printk("ERROR: fragment_bitcast sentinel i=%d got_i=%x got_f_bits=%x\n",
                       (int)i, out_i32[i], got_f);
            }
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t bits_dev = 0, f_dev = 0, out_i_dev = 0, out_f_dev = 0;
    if (vc4_m2_malloc(program, &bits_dev, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_malloc(program, &f_dev, LANES * sizeof(float)) < 0 ||
        vc4_m2_malloc(program, &out_i_dev, BUFFER_N * sizeof(uint32_t)) < 0 ||
        vc4_m2_malloc(program, &out_f_dev, BUFFER_N * sizeof(float)) < 0)
        panic("fragment_bitcast allocation failed");

    fill_buffers();
    int start = timer_get_usec();
    int launch_failures = 0;
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    if (vc4_m2_copy_htod(program, bits_dev, i_bits, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_htod(program, f_dev, f_values, LANES * sizeof(float)) < 0 ||
        vc4_m2_copy_htod(program, out_i_dev, out_i32, BUFFER_N * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_htod(program, out_f_dev, out_f32, BUFFER_N * sizeof(float)) < 0 ||
        fragment_bitcast_roundtrip_vc4kernel_launch(program, grid, block, bits_dev, f_dev, out_i_dev, out_f_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_i32, out_i_dev, BUFFER_N * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_dtoh(program, out_f32, out_f_dev, BUFFER_N * sizeof(float)) < 0)
        launch_failures++;

    int checksum_accum = 0;
    int total_mismatches = launch_failures ? 0 : verify_outputs(&checksum_accum);
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_bitcast_roundtrip_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d raw_word_oracle=1 checksum_accum=%d runtime_allocations=4 runtime_launches=1 elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, checksum_accum, elapsed);

    vc4Free(program, bits_dev);
    vc4Free(program, f_dev);
    vc4Free(program, out_i_dev);
    vc4Free(program, out_f_dev);
    vc4_program_destroy(program);
}
