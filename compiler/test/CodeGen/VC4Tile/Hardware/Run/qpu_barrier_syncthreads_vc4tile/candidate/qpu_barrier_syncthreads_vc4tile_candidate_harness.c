#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define QPU_BARRIER_SYNCTHREADS_VC4TILE_ACTIVE_QPUS 12u
#define QPU_BARRIER_SYNCTHREADS_VC4TILE_LANE_WIDTH 16u

void notmain(void) {
    struct vc4_program *program = 0;
    int launch_failures = 0;
    int start = timer_get_usec();

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("qpu_barrier_syncthreads_vc4tile vc4_program_create failed");

    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
    vc4_dim3 block = vc4_m2_dim3(QPU_BARRIER_SYNCTHREADS_VC4TILE_ACTIVE_QPUS * QPU_BARRIER_SYNCTHREADS_VC4TILE_LANE_WIDTH, 1u, 1u);

    printk("Running VC4 qpu_barrier_syncthreads_vc4tile candidate bundle...\n");
    if (qpu_barrier_syncthreads_vc4tile_launch(program, grid, block) < 0) {
        printk("ERROR: qpu_barrier_syncthreads_vc4tile launch failed\n");
        launch_failures++;
    }

    uint32_t runtime_launches = qpu_barrier_syncthreads_vc4tile_runtime_launches();
    int elapsed = timer_get_usec() - start;
    const char *status = (launch_failures == 0 && runtime_launches == 1u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=qpu_barrier_syncthreads_vc4tile status=%s runs=%d launch_failures=%d active_qpus=%d lanes=%d runtime_launches=%d elapsed_usec=%d\n",
           status, 1, launch_failures,
           (int)QPU_BARRIER_SYNCTHREADS_VC4TILE_ACTIVE_QPUS,
           (int)QPU_BARRIER_SYNCTHREADS_VC4TILE_LANE_WIDTH,
           (int)runtime_launches, elapsed);

    vc4_program_destroy(program);
}
