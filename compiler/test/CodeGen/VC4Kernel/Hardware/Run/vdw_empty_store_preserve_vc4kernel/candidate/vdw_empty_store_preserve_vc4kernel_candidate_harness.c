#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_ACTIVE_QPUS 1u
#define VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_LANES 16u
#define VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_BUFFER_N 64u
#define VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_SENTINEL 0xdeadbeefu

static uint32_t out_values[VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_BUFFER_N; i++)
        out_values[i] = VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_SENTINEL;
}

static int verify_all_sentinel(void) {
    int mismatches = 0;
    for (uint32_t i = 0; i < VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_BUFFER_N; i++) {
        if (out_values[i] != VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: vdw_empty_store_preserve_vc4kernel changed index=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i],
                       VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("vdw_empty_store_preserve_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 vdw_empty_store_preserve_vc4kernel candidate bundle...\n");
    fill_output();
    if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
        vdw_empty_store_preserve_vc4kernel_launch(program, grid, block, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
        printk("ERROR: vdw_empty_store_preserve_vc4kernel launch/copy failed\n");
        launch_failures++;
    } else {
        sentinel_mismatches = verify_all_sentinel();
        total_mismatches = sentinel_mismatches;
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=vdw_empty_store_preserve_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d buffer_n=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, 1, total_mismatches, sentinel_mismatches, launch_failures,
           VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_ACTIVE_QPUS,
           VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_LANES,
           VDW_EMPTY_STORE_PRESERVE_VC4KERNEL_BUFFER_N, 1, 1, elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
