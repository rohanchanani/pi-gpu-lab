#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define COND_BRANCH_SCALAR_GATE_VC4KERNEL_LANES 16u
#define COND_BRANCH_SCALAR_GATE_VC4KERNEL_ACTIVE_QPUS 1u
#define COND_BRANCH_SCALAR_GATE_VC4KERNEL_MAX_N 48u
#define COND_BRANCH_SCALAR_GATE_VC4KERNEL_GUARD 16u
#define COND_BRANCH_SCALAR_GATE_VC4KERNEL_BUFFER_N (COND_BRANCH_SCALAR_GATE_VC4KERNEL_MAX_N + COND_BRANCH_SCALAR_GATE_VC4KERNEL_GUARD)
#define COND_BRANCH_SCALAR_GATE_VC4KERNEL_SENTINEL 0xdeadbeefu
#define COND_BRANCH_SCALAR_GATE_VC4KERNEL_TAG 0x69000000u

struct cond_branch_scalar_gate_case {
    uint32_t control;
    uint32_t offset_elems;
};

static const struct cond_branch_scalar_gate_case cases[] = {
    {0u, 0u},
    {1u, 16u},
    {5u, 32u},
};

static uint32_t out_values[COND_BRANCH_SCALAR_GATE_VC4KERNEL_BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < COND_BRANCH_SCALAR_GATE_VC4KERNEL_BUFFER_N; i++)
        out_values[i] = COND_BRANCH_SCALAR_GATE_VC4KERNEL_SENTINEL;
}

static uint32_t expected_value(uint32_t control, uint32_t lane) {
    return COND_BRANCH_SCALAR_GATE_VC4KERNEL_TAG + (control << 8) + lane;
}

static int verify_case(const struct cond_branch_scalar_gate_case *tc) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < COND_BRANCH_SCALAR_GATE_VC4KERNEL_LANES; lane++) {
        uint32_t index = tc->offset_elems + lane;
        uint32_t expected = tc->control ? expected_value(tc->control, lane)
                                        : COND_BRANCH_SCALAR_GATE_VC4KERNEL_SENTINEL;
        if (out_values[index] != expected) {
            if (mismatches < 8)
                printk("ERROR: cond_branch_scalar_gate_vc4kernel control=%x index=%d gpu=%x expected=%x\n",
                       tc->control, (int)index, out_values[index], expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct cond_branch_scalar_gate_case *tc) {
    int mismatches = 0;
    uint32_t active_end = tc->offset_elems + COND_BRANCH_SCALAR_GATE_VC4KERNEL_LANES;
    for (uint32_t i = 0; i < COND_BRANCH_SCALAR_GATE_VC4KERNEL_BUFFER_N; i++) {
        if (tc->control && i >= tc->offset_elems && i < active_end)
            continue;
        if (out_values[i] != COND_BRANCH_SCALAR_GATE_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: cond_branch_scalar_gate_vc4kernel sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i],
                       COND_BRANCH_SCALAR_GATE_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(const struct cond_branch_scalar_gate_case *tc) {
    if (!tc->control)
        return 0;
    int checksum = 0;
    for (uint32_t lane = 0; lane < COND_BRANCH_SCALAR_GATE_VC4KERNEL_LANES; lane++)
        checksum += (int)(out_values[tc->offset_elems + lane] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = COND_BRANCH_SCALAR_GATE_VC4KERNEL_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("cond_branch_scalar_gate_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(COND_BRANCH_SCALAR_GATE_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 cond_branch_scalar_gate_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct cond_branch_scalar_gate_case *tc = &cases[case_id];
        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            cond_branch_scalar_gate_vc4kernel_launch(program, grid, block, out_dev,
                                                     tc->control, tc->offset_elems) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: cond_branch_scalar_gate_vc4kernel launch/copy failed case=%d control=%x\n",
                   (int)case_id, tc->control);
            launch_failures++;
            continue;
        }

        int mismatches = verify_case(tc);
        int sentinels = verify_sentinels(tc);
        int checksum = checksum_low16(tc);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        printk("COND_BRANCH_SCALAR_GATE_VC4KERNEL_CASE case=%d control=%x offset=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, tc->control, (int)tc->offset_elems, mismatches,
               sentinels, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=cond_branch_scalar_gate_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures,
           COND_BRANCH_SCALAR_GATE_VC4KERNEL_ACTIVE_QPUS,
           COND_BRANCH_SCALAR_GATE_VC4KERNEL_LANES,
           COND_BRANCH_SCALAR_GATE_VC4KERNEL_MAX_N, checksum_accum, 1,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
