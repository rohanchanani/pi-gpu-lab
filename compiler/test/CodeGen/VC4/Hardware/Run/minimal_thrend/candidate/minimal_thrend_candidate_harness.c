#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MINIMAL_THREND_ACTIVE_QPUS 12u

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(MINIMAL_THREND_ACTIVE_QPUS, 1, 1);

    printk("Running VC4 minimal_thrend M2 candidate bundle...\n");
    int start = timer_get_usec();
    int launch_failures = 0;
    if (minimal_thrend_launch(program, grid, block) < 0)
        launch_failures++;
    int elapsed = timer_get_usec() - start;
    uint32_t runtime_allocations = minimal_thrend_runtime_allocations();
    uint32_t runtime_launches = minimal_thrend_runtime_launches();

    const char *status = launch_failures == 0 ? "PASS" : "FAIL";
    uint32_t completed_qpus = launch_failures == 0 ? MINIMAL_THREND_ACTIVE_QPUS : 0u;
    printk("VC4_TEST_RESULT name=minimal_thrend status=%s completed_qpus=%d launch_failures=%d active_qpus=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)completed_qpus, launch_failures, (int)MINIMAL_THREND_ACTIVE_QPUS,
           (int)runtime_allocations, (int)runtime_launches, elapsed);
    if (launch_failures)
        panic("minimal_thrend launch failed: launch_failures=%d", launch_failures);

    vc4_program_destroy(program);
}
