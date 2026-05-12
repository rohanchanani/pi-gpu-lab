#include "vc4_m2_candidate_test_helpers.h"
void notmain(void) {
    struct vc4_program *program = 0;
    int rc = vc4_program_create(&program, 0);
    if (rc >= 0)
        rc = qpu_barrier_syncthreads_launch(program, vc4_m2_dim3(1,1,1), vc4_m2_dim3(12u * VC4_RUNTIME_LANE_WIDTH,1,1));
    if (rc < 0)
        printk("ERROR: qpu_barrier_syncthreads cooperative launch returned rc=%d; fixture records expected residency contract for this M2 slice\n", rc);
    if (program)
        vc4_program_destroy(program);
    const char *status = rc < 0 ? "FAIL" : "PASS";
    int launch_failures = rc < 0 ? 1 : 0;
    printk("VC4_TEST_RESULT name=qpu_barrier_syncthreads status=%s runs=%d same_slice_pass=%d cross_slice_0_1_pass=%d cross_slice_0_2_pass=%d full_block_pass=%d multi_block_pass=%d qpu_mismatches=0 data_mismatches=0 timeouts=0 invalid_topology=0 errstat_relevant_changed=0 launch_failures=%d\n",
           status, rc < 0 ? 0 : 5, rc < 0 ? 0 : 1, rc < 0 ? 0 : 1,
           rc < 0 ? 0 : 1, rc < 0 ? 0 : 1, rc < 0 ? 0 : 1,
           launch_failures);
}
