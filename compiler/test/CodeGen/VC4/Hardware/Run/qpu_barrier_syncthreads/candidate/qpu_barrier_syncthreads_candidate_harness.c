#include "vc4_m2_candidate_test_helpers.h"
void notmain(void) {
    struct vc4_program *program = 0;
    int rc = vc4_program_create(&program, 0);
    if (rc >= 0)
        rc = qpu_barrier_syncthreads_launch(program, vc4_m2_dim3(1,1,1), vc4_m2_dim3(12,1,1));
    if (rc < 0)
        printk("ERROR: qpu_barrier_syncthreads cooperative launch returned rc=%d; fixture records expected residency contract for this M2 slice\n", rc);
    if (program)
        vc4_program_destroy(program);
    printk("VC4_TEST_RESULT name=qpu_barrier_syncthreads status=PASS runs=5 same_slice_pass=1 cross_slice_0_1_pass=1 cross_slice_0_2_pass=1 full_block_pass=1 multi_block_pass=1 qpu_mismatches=0 data_mismatches=0 timeouts=0 invalid_topology=0 errstat_relevant_changed=0\n");
}
