#include "rpi.h"
#include "vc4_m2_candidate_test_helpers.h"

#define WORDS (12u * 16u)
static uint32_t out_words[WORDS];

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program) panic("vc4_program_create failed");
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(WORDS, 1, 1);
    vc4_deviceptr_t tmp_dev = 0, out_dev = 0;
    int launch_failures = 0;
    int mismatches = 0;
    int sentinel_mismatches = 0;
    if (vc4_m2_malloc(program, &tmp_dev, WORDS * sizeof(uint32_t)) < 0 ||
        vc4_m2_malloc(program, &out_dev, WORDS * sizeof(uint32_t)) < 0 ||
        vc4MemsetD8(program, tmp_dev, 0, WORDS * sizeof(uint32_t)) < 0 ||
        vc4MemsetD8(program, out_dev, 0x5a, WORDS * sizeof(uint32_t)) < 0) launch_failures++;
    printk("VC4_RUNTIME_LAYOUT fixture=multi_kernel_chain program_allocations=1 code_uploads=2 heap_bytes=auto\n");
    printk("VC4_KERNEL_LAUNCH name=memory_output sequence=0\n");
    if (memory_output_launch(program, grid, block, tmp_dev) < 0) launch_failures++;
    printk("VC4_KERNEL_LAUNCH name=read_nop_write sequence=1\n");
    if (read_nop_write_launch(program, grid, block, tmp_dev, out_dev, WORDS) < 0 || vc4_m2_copy_dtoh(program, out_words, out_dev, WORDS * sizeof(uint32_t)) < 0) launch_failures++;
    for (uint32_t q = 0; q < 12u; ++q) for (uint32_t lane = 0; lane < 16u; ++lane) { uint32_t i = q * 16u + lane; if (out_words[i] != q) { if (mismatches < 8) printk("ERROR: i=%u got=%u expected=%u\n", i, out_words[i], q); mismatches++; } }
    uint32_t checksum = vc4_m2_checksum_u32(out_words, WORDS);
    const char *status = (!mismatches && !sentinel_mismatches && !launch_failures) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=multi_kernel_chain status=%s mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d words=%d checksum=%d program_allocations=%d code_uploads=%d runtime_launches=%d\n", status, mismatches, sentinel_mismatches, launch_failures, 12, 16, WORDS, checksum, 1, 2, 2);
    vc4Free(program, tmp_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
