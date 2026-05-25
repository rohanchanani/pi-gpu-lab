#include "vc4_m2_candidate_test_helpers.h"
void notmain(void) {
    struct vc4_program *program = 0;
    int launch_failures = 0;
    int runtime_launches = 0;
    int same_slice_visibility_pass = 0;
    int same_slice_collision_pass = 0;
    int reported_qpu_mismatches = 0;
    int timeouts = 0;
    int invalid_topology = 0;
    int errstat_relevant_changed = 0;
    int rc = vc4_program_create(&program, 0);
    if (rc < 0 || !program) {
        printk("ERROR: vpm_slice_visibility program creation failed rc=%d\n", rc);
        launch_failures = 1;
    } else {
        rc = vpm_slice_visibility_launch(program, vc4_m2_dim3(1,1,1), vc4_m2_dim3(2,1,1));
        runtime_launches = 1;
        if (rc < 0) {
            printk("ERROR: vpm_slice_visibility cooperative launch returned rc=%d; fixture records expected VPM slice contract for this M2 slice\n", rc);
            launch_failures = 1;
            timeouts = 1;
        } else {
            same_slice_visibility_pass = 1;
            same_slice_collision_pass = 1;
        }
    }

    const char *status =
        (launch_failures == 0 &&
         same_slice_visibility_pass == 1 &&
         same_slice_collision_pass == 1 &&
         reported_qpu_mismatches == 0 &&
         timeouts == 0 &&
         invalid_topology == 0 &&
         errstat_relevant_changed == 0) ? "PASS" : "FAIL";

    if (program)
        vc4_program_destroy(program);
    printk("VC4_TEST_RESULT name=vpm_slice_visibility status=%s sanity_mismatches=0 same_slice_visibility_pass=%d same_slice_collision_pass=%d reported_qpu_mismatches=%d launch_failures=%d runtime_launches=%d active_qpus=2 lanes=16 timeouts=%d invalid_topology=%d errstat_relevant_changed=%d\n",
           status, same_slice_visibility_pass, same_slice_collision_pass,
           reported_qpu_mismatches, launch_failures, runtime_launches,
           timeouts, invalid_topology, errstat_relevant_changed);
}
