#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_ACTIVE_QPUS 1u
#define GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_LANES 16u
#define GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_GUARD 16u
#define GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_BUFFER_N (GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_LANES + GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_GUARD)
#define GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_SENTINEL 0xdeadbeefu
#define GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_OLD_TAG 0x59000000u
#define GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_NEW_TAG 0x54000000u

static const uint32_t alternating_mask[GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_LANES] = {
    1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u,
    1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u
};

static const uint32_t irregular_mask[GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_LANES] = {
    1u, 1u, 0u, 1u, 0u, 0u, 1u, 0u,
    1u, 0u, 0u, 0u, 1u, 1u, 0u, 1u
};

static uint32_t mask_values[GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_BUFFER_N];
static uint32_t out_values[GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_BUFFER_N];

static void fill_buffers(const uint32_t *mask) {
    for (uint32_t i = 0; i < GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_BUFFER_N; i++) {
        mask_values[i] = 0u;
        out_values[i] = i < GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_LANES
                            ? GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_OLD_TAG + i
                            : GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_SENTINEL;
    }
    for (uint32_t i = 0; i < GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_LANES; i++)
        mask_values[i] = mask[i];
}

static int verify_values(const uint32_t *mask) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_LANES; lane++) {
        uint32_t expected = mask[lane] == 1u
                                ? GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_NEW_TAG + lane
                                : GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_OLD_TAG + lane;
        if (out_values[lane] != expected) {
            if (mismatches < 8)
                printk("ERROR: general_mask_store lane=%d mask=%d gpu=%x expected=%x\n",
                       (int)lane, (int)mask[lane], out_values[lane], expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_LANES;
         i < GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_BUFFER_N; i++) {
        if (out_values[i] != GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: general_mask_store sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i],
                       GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(void) {
    int checksum = 0;
    for (uint32_t i = 0; i < GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_LANES; i++)
        checksum += (int)(out_values[i] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    uint32_t bytes = GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_BUFFER_N * sizeof(uint32_t);
    vc4_deviceptr_t mask_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &mask_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("general_mask_store_preserve_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int alternating_cases = 0;
    int irregular_cases = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 general_mask_store_preserve_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < 2u; case_id++) {
        const uint32_t *mask = case_id == 0u ? alternating_mask : irregular_mask;
        fill_buffers(mask);
        if (vc4_m2_copy_htod(program, mask_dev, mask_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            general_mask_store_preserve_vc4kernel_launch(program, grid, block, mask_dev, out_dev) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: general_mask_store_preserve_vc4kernel launch/copy failed case=%d\n",
                   (int)case_id);
            launch_failures++;
            continue;
        }

        int mismatches = verify_values(mask);
        int sentinels = verify_sentinels();
        int checksum = checksum_low16();
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        if (case_id == 0u)
            alternating_cases++;
        else
            irregular_cases++;
        printk("GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_CASE case=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, mismatches, sentinels, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && alternating_cases == 1 &&
                          irregular_cases == 1) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=general_mask_store_preserve_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d alternating_cases=%d irregular_cases=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, 2, total_mismatches, sentinel_mismatches, launch_failures,
           GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_ACTIVE_QPUS,
           GENERAL_MASK_STORE_PRESERVE_VC4KERNEL_LANES, alternating_cases,
           irregular_cases, checksum_accum, 2, 2, elapsed);

    vc4Free(program, mask_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
