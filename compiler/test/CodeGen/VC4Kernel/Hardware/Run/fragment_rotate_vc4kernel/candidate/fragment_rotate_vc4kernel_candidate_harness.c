#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define FRAGMENT_ROTATE_VC4KERNEL_LANES 16u
#define FRAGMENT_ROTATE_VC4KERNEL_ACTIVE_QPUS 1u
#define FRAGMENT_ROTATE_VC4KERNEL_ROTATE_AMOUNT 3u
#define FRAGMENT_ROTATE_VC4KERNEL_MAX_N 32u
#define FRAGMENT_ROTATE_VC4KERNEL_GUARD 16u
#define FRAGMENT_ROTATE_VC4KERNEL_BUFFER_N (FRAGMENT_ROTATE_VC4KERNEL_MAX_N + FRAGMENT_ROTATE_VC4KERNEL_GUARD)
#define FRAGMENT_ROTATE_VC4KERNEL_SENTINEL 0xdeadbeefu

struct fragment_rotate_case {
    uint32_t offset_elems;
    uint32_t base_value;
};

static const struct fragment_rotate_case cases[] = {
    {0u, 0x42000000u},
    {16u, 0x43000000u},
};

static uint32_t out_values[FRAGMENT_ROTATE_VC4KERNEL_BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < FRAGMENT_ROTATE_VC4KERNEL_BUFFER_N; i++)
        out_values[i] = FRAGMENT_ROTATE_VC4KERNEL_SENTINEL;
}

static uint32_t expected_value(uint32_t base_value, uint32_t lane) {
    uint32_t source_lane = (lane + FRAGMENT_ROTATE_VC4KERNEL_ROTATE_AMOUNT) &
                           (FRAGMENT_ROTATE_VC4KERNEL_LANES - 1u);
    return base_value + source_lane;
}

static int verify_active(uint32_t offset_elems, uint32_t base_value) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < FRAGMENT_ROTATE_VC4KERNEL_LANES; lane++) {
        uint32_t index = offset_elems + lane;
        uint32_t expected = expected_value(base_value, lane);
        if (out_values[index] != expected) {
            if (mismatches < 8)
                printk("ERROR: fragment_rotate lane=%d index=%d gpu=%x expected=%x\n",
                       (int)lane, (int)index, out_values[index], expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t offset_elems) {
    int mismatches = 0;
    uint32_t active_end = offset_elems + FRAGMENT_ROTATE_VC4KERNEL_LANES;
    for (uint32_t i = 0; i < FRAGMENT_ROTATE_VC4KERNEL_BUFFER_N; i++) {
        if (i >= offset_elems && i < active_end)
            continue;
        if (out_values[i] != FRAGMENT_ROTATE_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_rotate sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], FRAGMENT_ROTATE_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(uint32_t offset_elems) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < FRAGMENT_ROTATE_VC4KERNEL_LANES; lane++)
        checksum += (int)(out_values[offset_elems + lane] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = FRAGMENT_ROTATE_VC4KERNEL_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("fragment_rotate_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(FRAGMENT_ROTATE_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 fragment_rotate_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t offset_elems = cases[case_id].offset_elems;
        uint32_t base_value = cases[case_id].base_value;
        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            fragment_rotate_vc4kernel_launch(program, grid, block, out_dev, offset_elems, base_value) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: fragment_rotate_vc4kernel launch/copy failed case=%d\n",
                   (int)case_id);
            launch_failures++;
            continue;
        }

        int mismatches = verify_active(offset_elems, base_value);
        int sentinels = verify_sentinels(offset_elems);
        int checksum = checksum_low16(offset_elems);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        printk("FRAGMENT_ROTATE_VC4KERNEL_CASE case=%d offset=%d base=%x rotate_amount=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)offset_elems, base_value,
               FRAGMENT_ROTATE_VC4KERNEL_ROTATE_AMOUNT, mismatches, sentinels,
               checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_rotate_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d rotate_amount=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures,
           FRAGMENT_ROTATE_VC4KERNEL_ACTIVE_QPUS, FRAGMENT_ROTATE_VC4KERNEL_LANES,
           FRAGMENT_ROTATE_VC4KERNEL_ROTATE_AMOUNT, checksum_accum, 1,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
