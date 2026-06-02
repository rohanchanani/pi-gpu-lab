#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_ACTIVE_QPUS 1u
#define TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_LANES 16u
#define TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_GUARD 16u
#define TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_BUFFER_N (TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_LANES + TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_GUARD)
#define TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_SENTINEL 0xdeadbeefu
#define TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_TAG 0x73000000u

static const uint32_t cases[] = {0u, 1u, 5u, 9u, 16u};
static uint32_t input_values[TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_BUFFER_N];
static uint32_t out_values[TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_BUFFER_N];

static void fill_buffers(void) {
    for (uint32_t i = 0; i < TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_BUFFER_N; i++) {
        input_values[i] = TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_TAG + i;
        out_values[i] = TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_SENTINEL;
    }
}

static int verify_values(uint32_t threshold) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_LANES; lane++) {
        uint32_t expected = lane < threshold ? input_values[lane] : 0u;
        if (out_values[lane] != expected) {
            if (mismatches < 8)
                printk("ERROR: tmu_general_mask threshold=%d lane=%d gpu=%x expected=%x\n",
                       (int)threshold, (int)lane, out_values[lane], expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_LANES;
         i < TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_BUFFER_N; i++) {
        if (out_values[i] != TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: tmu_general_mask sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i],
                       TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(void) {
    int checksum = 0;
    for (uint32_t i = 0; i < TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_LANES; i++)
        checksum += (int)(out_values[i] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    uint32_t bytes = TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_BUFFER_N * sizeof(uint32_t);
    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &input_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("tmu_general_mask_load_zero_fill_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_threshold0 = 0;
    int saw_threshold9 = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 tmu_general_mask_load_zero_fill_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t threshold = cases[case_id];
        fill_buffers();
        if (vc4_m2_copy_htod(program, input_dev, input_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            tmu_general_mask_load_zero_fill_vc4kernel_launch(program, grid, block, input_dev, out_dev, threshold) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: tmu_general_mask_load_zero_fill_vc4kernel launch/copy failed case=%d threshold=%d\n",
                   (int)case_id, (int)threshold);
            launch_failures++;
            continue;
        }

        int mismatches = verify_values(threshold);
        int sentinels = verify_sentinels();
        int checksum = checksum_low16();
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        if (threshold == 0u)
            saw_threshold0 = 1;
        if (threshold == 9u)
            saw_threshold9 = 1;
        printk("TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_CASE case=%d threshold=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)threshold, mismatches, sentinels, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_threshold0 &&
                          saw_threshold9) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=tmu_general_mask_load_zero_fill_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d saw_threshold0=%d saw_threshold9=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures,
           TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_ACTIVE_QPUS,
           TMU_GENERAL_MASK_LOAD_ZERO_FILL_VC4KERNEL_LANES, saw_threshold0,
           saw_threshold9, checksum_accum, 2,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
