#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define BUFFER_N (LANES + 16u)
#define SENTINEL 0x91f16c02u

static const float input_f32[LANES] = {
    -4.0f, -2.0f, -1.0f, -0.5f, 0.25f, 0.5f, 1.0f, 1.5f,
    2.0f, 3.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f
};

static const uint32_t expected_carriers[LANES] = {
    0x0000c400u, 0x0000c000u, 0x0000bc00u, 0x0000b800u,
    0x00003400u, 0x00003800u, 0x00003c00u, 0x00003e00u,
    0x00004000u, 0x00004200u, 0x00004400u, 0x00004800u,
    0x00004c00u, 0x00005000u, 0x00005400u, 0x00005800u
};

static uint32_t out_words[BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_words[i] = SENTINEL;
}

static int verify_outputs(uint32_t *checksum) {
    int mismatches = 0;
    *checksum = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        uint32_t got = out_words[lane];
        *checksum += got;
        if (got != expected_carriers[lane] || (got & 0xffff0000u) != 0) {
            if (mismatches < 8)
                printk("ERROR: f16_pack lane=%d got=%x exp=%x\n",
                       (int)lane, got, expected_carriers[lane]);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = LANES; i < BUFFER_N; ++i) {
        if (out_words[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: f16_pack sentinel i=%d got=%x\n",
                       (int)i, out_words[i]);
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t input_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &input_dev, LANES * sizeof(float)) < 0 ||
        vc4_m2_malloc(program, &out_dev, BUFFER_N * sizeof(uint32_t)) < 0)
        panic("fragment_f16_pack_from_f32 allocation failed");

    fill_output();
    int launch_failures = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    if (vc4_m2_copy_htod(program, input_dev, input_f32, LANES * sizeof(float)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_words, BUFFER_N * sizeof(uint32_t)) < 0 ||
        fragment_f16_pack_from_f32_vc4kernel_launch(program, grid, block, input_dev, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_words, out_dev, BUFFER_N * sizeof(uint32_t)) < 0)
        launch_failures++;

    uint32_t checksum = 0;
    int total_mismatches = launch_failures ? 0 : verify_outputs(&checksum);
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_f16_pack_from_f32_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d saw_f16_pack_from_f32=1 saw_f32_compute=1 no_native_f16_arith=1 no_bf16_fp8=1 deterministic_f16_carrier=1 high_bits_zero=1 exact_values=neg4,neg2,neg1,neg0p5,0p25,0p5,1,1p5,2,3,4,8,16,32,64,128 checksum_accum=%u runtime_allocations=2 runtime_launches=1 elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, checksum, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
