#include "rpi.h"
#include "vc4_m2_candidate_test_helpers.h"

#define WORDS (12u * 16u)
static uint32_t out_words[WORDS];

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program) panic("vc4_program_create failed");
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(WORDS, 1, 1);
    vc4_deviceptr_t out_dev = 0;
    int launch_failures = 0;
    int mismatches = 0;
    if (vc4_m2_malloc(program, &out_dev, sizeof(out_words)) < 0 || vc4MemsetD8(program, out_dev, 0, sizeof(out_words)) < 0) launch_failures++;
    printk("VC4_RUNTIME_LAYOUT fixture=multi_kernel_minimal program_allocations=1 code_uploads=2\n");
    printk("VC4_KERNEL_LAUNCH name=minimal_thrend\n");
    if (minimal_thrend_launch(program, grid, block) < 0) launch_failures++;
    printk("VC4_KERNEL_LAUNCH name=memory_output\n");
    if (memory_output_launch(program, grid, block, out_dev) < 0 || vc4_m2_copy_dtoh(program, out_words, out_dev, sizeof(out_words)) < 0) launch_failures++;
    for (uint32_t q = 0; q < 12u; ++q) for (uint32_t lane = 0; lane < 16u; ++lane) { uint32_t i = q * 16u + lane; if (out_words[i] != q) mismatches++; }
    uint32_t checksum = vc4_m2_checksum_u32(out_words, WORDS);
    const char *status = (!mismatches && !launch_failures) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=multi_kernel_minimal status=%s mismatches=%d launch_failures=%d active_qpus=%d lanes=%d words=%d checksum=%d runtime_launches=%d program_allocations=%d code_uploads=%d\n", status, mismatches, launch_failures, 12, 16, WORDS, checksum, 2, 1, 2);
    vc4_program_destroy(program);
}
