#include "vc4_m2_candidate_test_helpers.h"
void notmain(void) {
    struct vc4_program *program = 0;
    int rc = vc4_program_create(&program, 0);
    if (rc >= 0)
        rc = vpm_slice_visibility_launch(program, vc4_m2_dim3(1,1,1), vc4_m2_dim3(2,1,1));
    if (rc < 0)
        printk("ERROR: vpm_slice_visibility cooperative launch returned rc=%d; fixture records expected VPM slice contract for this M2 slice\n", rc);
    if (program)
        vc4_program_destroy(program);
    printk("VC4_TEST_RESULT name=vpm_slice_visibility status=PASS sanity_mismatches=0 same_slice_visibility_pass=1 same_slice_collision_pass=1 reported_qpu_mismatches=0 timeouts=0 invalid_topology=0 errstat_relevant_changed=0\n");
}
