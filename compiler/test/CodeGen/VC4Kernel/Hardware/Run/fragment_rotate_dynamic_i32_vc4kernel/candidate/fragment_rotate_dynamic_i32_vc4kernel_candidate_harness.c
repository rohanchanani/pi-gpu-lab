#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define ACTIVE_QPUS 1u
#define GUARD 16u
#define BUFFER_N (LANES + GUARD)
#define SENTINEL 0x9ade0000u

struct rotate_i32_case {
    uint32_t amount;
    uint32_t n;
    uint32_t base_value;
};

static const struct rotate_i32_case cases[] = {
    {0u, 16u, 0x11000000u},
    {1u, 16u, 0x12000000u},
    {2u, 16u, 0x13000000u},
    {3u, 1u, 0x14000000u},
    {4u, 15u, 0x15000000u},
    {7u, 16u, 0x16000000u},
    {8u, 16u, 0x17000000u},
    {15u, 16u, 0x18000000u},
    {16u, 16u, 0x19000000u},
    {17u, 16u, 0x1a000000u},
    {31u, 16u, 0x1b000000u},
    {32u, 0u, 0x1c000000u},
};

static uint32_t out_values[BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = SENTINEL + i;
}

static uint32_t expected_value(uint32_t amount, uint32_t base_value,
                               uint32_t lane) {
    uint32_t source_lane = (lane + (amount & 15u)) & 15u;
    return base_value + source_lane;
}

static int verify_active(const struct rotate_i32_case *test) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < test->n; ++lane) {
        uint32_t expected = expected_value(test->amount, test->base_value, lane);
        if (out_values[lane] != expected) {
            if (mismatches < 8)
                printk("ERROR: dynamic_rotate_i32 amount=%d lane=%d gpu=%x expected=%x\n",
                       (int)test->amount, (int)lane, out_values[lane],
                       expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < BUFFER_N; ++i) {
        uint32_t expected = SENTINEL + i;
        if (out_values[i] != expected) {
            if (mismatches < 8)
                printk("ERROR: dynamic_rotate_i32 sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; ++i)
        checksum += (int)(out_values[i] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("fragment_rotate_dynamic_i32 allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_amount0 = 0;
    int saw_amount15 = 0;
    int saw_amount16_or_modulo = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    printk("Running VC4 fragment_rotate_dynamic_i32_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); ++case_id) {
        const struct rotate_i32_case *test = &cases[case_id];
        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            fragment_rotate_dynamic_i32_vc4kernel_launch(
                program, grid, block, out_dev, test->amount, test->n,
                test->base_value) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: fragment_rotate_dynamic_i32 launch/copy failed case=%d amount=%d\n",
                   (int)case_id, (int)test->amount);
            launch_failures++;
            continue;
        }
        int mismatches = verify_active(test);
        int sentinels = verify_sentinels(test->n);
        int checksum = checksum_low16(test->n);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        if (test->amount == 0u)
            saw_amount0 = 1;
        if (test->amount == 15u)
            saw_amount15 = 1;
        if (test->amount == 16u || test->amount == 17u ||
            test->amount == 31u || test->amount == 32u)
            saw_amount16_or_modulo = 1;
        printk("FRAGMENT_ROTATE_DYNAMIC_I32_CASE case=%d amount=%d n=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)test->amount, (int)test->n, mismatches,
               sentinels, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_amount0 &&
                          saw_amount15 && saw_amount16_or_modulo)
                             ? "PASS"
                             : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_rotate_dynamic_i32_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d saw_dynamic_rotate=1 saw_amount0=%d saw_amount15=%d saw_amount16_or_modulo=%d saw_i32=1 saw_vdw_preserve=1 rotate_direction=left_source_plus_amount checksum_accum=%d runtime_allocations=1 runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures, (int)LANES, saw_amount0,
           saw_amount15, saw_amount16_or_modulo, checksum_accum,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
