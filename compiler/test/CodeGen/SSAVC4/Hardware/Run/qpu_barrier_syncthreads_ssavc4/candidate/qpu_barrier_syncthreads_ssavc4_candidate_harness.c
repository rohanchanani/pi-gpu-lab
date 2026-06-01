#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define QPU_BARRIER_SYNCTHREADS_SSAVC4_ACTIVE_QPUS 12u
#define QPU_BARRIER_SYNCTHREADS_SSAVC4_LANE_WIDTH 16u

void notmain(void) {
    struct vc4_program *program = 0;
    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int start = timer_get_usec();

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("qpu_barrier_syncthreads_ssavc4 vc4_program_create failed");

    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
    vc4_dim3 block = vc4_m2_dim3(QPU_BARRIER_SYNCTHREADS_SSAVC4_ACTIVE_QPUS * QPU_BARRIER_SYNCTHREADS_SSAVC4_LANE_WIDTH, 1u, 1u);

    printk("Running VC4 qpu_barrier_syncthreads_ssavc4 candidate bundle...\n");
    if (qpu_barrier_syncthreads_ssavc4_launch(program, grid, block) < 0) {
        printk("ERROR: qpu_barrier_syncthreads_ssavc4 launch failed\n");
        launch_failures++;
    }

    int elapsed = timer_get_usec() - start;
    const char *status =
        (total_mismatches == 0 && sentinel_mismatches == 0 &&
         launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=qpu_barrier_syncthreads_ssavc4 status=%s runs=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d runtime_launches=%d elapsed_usec=%d\n",
           status, 1, total_mismatches, sentinel_mismatches, launch_failures,
           (int)QPU_BARRIER_SYNCTHREADS_SSAVC4_ACTIVE_QPUS,
           (int)QPU_BARRIER_SYNCTHREADS_SSAVC4_LANE_WIDTH, 1, elapsed);

    vc4_program_destroy(program);
}
