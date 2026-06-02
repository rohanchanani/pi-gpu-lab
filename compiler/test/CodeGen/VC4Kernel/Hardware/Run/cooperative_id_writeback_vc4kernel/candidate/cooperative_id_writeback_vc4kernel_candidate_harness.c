#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define COOPERATIVE_ID_WRITEBACK_VC4KERNEL_WARPS 12u
#define COOPERATIVE_ID_WRITEBACK_VC4KERNEL_LANES 16u
#define COOPERATIVE_ID_WRITEBACK_VC4KERNEL_WORDS (COOPERATIVE_ID_WRITEBACK_VC4KERNEL_WARPS * COOPERATIVE_ID_WRITEBACK_VC4KERNEL_LANES)
#define COOPERATIVE_ID_WRITEBACK_VC4KERNEL_GUARD 16u
#define COOPERATIVE_ID_WRITEBACK_VC4KERNEL_SENTINEL 0x6e7f8091u
#define COOPERATIVE_ID_WRITEBACK_VC4KERNEL_TAG 0x63000000u

static uint32_t output_values[COOPERATIVE_ID_WRITEBACK_VC4KERNEL_WORDS + COOPERATIVE_ID_WRITEBACK_VC4KERNEL_GUARD];

static uint32_t checksum_words(const uint32_t *values, uint32_t count) {
    uint32_t hash = 2166136261u;
    for (uint32_t i = 0; i < count; i++) {
        hash ^= values[i];
        hash *= 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t output_dev = 0;
    uint32_t output_bytes = sizeof(output_values);
    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int start = timer_get_usec();

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("cooperative_id_writeback_vc4kernel vc4_program_create failed");
    if (vc4Malloc(program, &output_dev, output_bytes) < 0)
        panic("cooperative_id_writeback_vc4kernel output allocation failed");

    for (uint32_t i = 0; i < COOPERATIVE_ID_WRITEBACK_VC4KERNEL_WORDS + COOPERATIVE_ID_WRITEBACK_VC4KERNEL_GUARD; i++)
        output_values[i] = COOPERATIVE_ID_WRITEBACK_VC4KERNEL_SENTINEL;

    if (vc4MemcpyHtoD(program, output_dev, output_values, output_bytes) < 0) {
        launch_failures++;
    } else {
        vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
        vc4_dim3 block = vc4_m2_dim3(COOPERATIVE_ID_WRITEBACK_VC4KERNEL_WORDS, 1u, 1u);
        if (cooperative_id_writeback_vc4kernel_launch(program, grid, block, output_dev) < 0 ||
            vc4MemcpyDtoH(program, output_values, output_dev, output_bytes) < 0) {
            printk("ERROR: cooperative_id_writeback_vc4kernel launch/copy failed\n");
            launch_failures++;
        }
    }

    for (uint32_t warp = 0; warp < COOPERATIVE_ID_WRITEBACK_VC4KERNEL_WARPS; warp++) {
        for (uint32_t lane = 0; lane < COOPERATIVE_ID_WRITEBACK_VC4KERNEL_LANES; lane++) {
            uint32_t index = warp * COOPERATIVE_ID_WRITEBACK_VC4KERNEL_LANES + lane;
            uint32_t expected = COOPERATIVE_ID_WRITEBACK_VC4KERNEL_TAG | (warp << 4) | lane;
            uint32_t got = output_values[index];
            if (got != expected) {
                if (total_mismatches < 8)
                    printk("ERROR: warp=%d lane=%d got=%x expected=%x\n",
                           (int)warp, (int)lane, got, expected);
                total_mismatches++;
            }
        }
    }

    for (uint32_t i = 0; i < COOPERATIVE_ID_WRITEBACK_VC4KERNEL_GUARD; i++) {
        uint32_t got = output_values[COOPERATIVE_ID_WRITEBACK_VC4KERNEL_WORDS + i];
        if (got != COOPERATIVE_ID_WRITEBACK_VC4KERNEL_SENTINEL) {
            if (sentinel_mismatches < 8)
                printk("ERROR: guard=%d got=%x expected=%x\n",
                       (int)i, got, COOPERATIVE_ID_WRITEBACK_VC4KERNEL_SENTINEL);
            sentinel_mismatches++;
        }
    }

    uint32_t checksum = checksum_words(output_values, COOPERATIVE_ID_WRITEBACK_VC4KERNEL_WORDS);
    int elapsed = timer_get_usec() - start;
    const char *status =
        (total_mismatches == 0 && sentinel_mismatches == 0 &&
         launch_failures == 0) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=cooperative_id_writeback_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d warps_per_block=%d runtime_launches=%d checksum=%x elapsed_usec=%d\n",
           status, 1, total_mismatches, sentinel_mismatches, launch_failures,
           (int)COOPERATIVE_ID_WRITEBACK_VC4KERNEL_WARPS,
           (int)COOPERATIVE_ID_WRITEBACK_VC4KERNEL_LANES,
           (int)COOPERATIVE_ID_WRITEBACK_VC4KERNEL_WARPS, 1, checksum, elapsed);

    vc4Free(program, output_dev);
    vc4_program_destroy(program);
}
