#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_LANES 16u
#define CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_ACTIVE_QPUS 1u
#define CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_MAX_N 64u
#define CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_GUARD 16u
#define CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_BUFFER_N (CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_MAX_N + CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_GUARD)
#define CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_SENTINEL 0xdeadbeefu
#define CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_ALL_BASE 0x6b001000u
#define CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_PARTIAL_BASE 0x6b002000u
#define CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_EMPTY_BASE 0x6b003000u

struct control_flow_tail_block_args_case {
    uint32_t n;
    uint32_t offset_elems;
    uint32_t expected_base;
};

static const struct control_flow_tail_block_args_case cases[] = {
    {0u, 0u, CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_EMPTY_BASE},
    {5u, 16u, CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_PARTIAL_BASE},
    {16u, 32u, CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_ALL_BASE},
    {17u, 48u, CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_ALL_BASE},
};

static uint32_t out_values[CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_BUFFER_N; i++)
        out_values[i] = CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_SENTINEL;
}

static int verify_active(const struct control_flow_tail_block_args_case *tc) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_LANES; lane++) {
        uint32_t index = tc->offset_elems + lane;
        uint32_t expected = tc->expected_base + lane;
        if (out_values[index] != expected) {
            if (mismatches < 8)
                printk("ERROR: control_flow_tail_block_args n=%d index=%d gpu=%x expected=%x\n",
                       (int)tc->n, (int)index, out_values[index], expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t offset_elems) {
    int mismatches = 0;
    uint32_t active_end = offset_elems + CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_LANES;
    for (uint32_t i = 0; i < CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_BUFFER_N; i++) {
        if (i >= offset_elems && i < active_end)
            continue;
        if (out_values[i] != CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: control_flow_tail_block_args sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i],
                       CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(uint32_t offset_elems) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_LANES; lane++)
        checksum += (int)(out_values[offset_elems + lane] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("control_flow_tail_block_args_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 control_flow_tail_block_args_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct control_flow_tail_block_args_case *tc = &cases[case_id];
        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            control_flow_tail_block_args_vc4kernel_launch(
                program, grid, block, out_dev, tc->n, tc->offset_elems) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: control_flow_tail_block_args_vc4kernel launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)tc->n);
            launch_failures++;
            continue;
        }

        int mismatches = verify_active(tc);
        int sentinels = verify_sentinels(tc->offset_elems);
        int checksum = checksum_low16(tc->offset_elems);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        printk("CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_CASE case=%d n=%d offset=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)tc->n, (int)tc->offset_elems, mismatches,
               sentinels, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=control_flow_tail_block_args_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures,
           CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_ACTIVE_QPUS,
           CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_LANES,
           CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4KERNEL_MAX_N, checksum_accum, 1,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
