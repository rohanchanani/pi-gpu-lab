#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define WORDS (12u * 16u)
#define GUARD_WORDS 16u
#define SENTINEL_BASE 0xdead0000u
static uint32_t tmp_words[WORDS + GUARD_WORDS];
static uint32_t out_words[WORDS + GUARD_WORDS];

static void fill_buffers(void) {
    for (uint32_t i = 0; i < WORDS + GUARD_WORDS; ++i) {
        tmp_words[i] = SENTINEL_BASE | i;
        out_words[i] = SENTINEL_BASE | (0x100u + i);
    }
}

static int verify_pattern(const char *label, const uint32_t *values) {
    int mismatches = 0;
    for (uint32_t q = 0; q < 12u; ++q) {
        for (uint32_t lane = 0; lane < 16u; ++lane) {
            uint32_t i = q * 16u + lane;
            if (values[i] != q) {
                if (mismatches < 8)
                    printk("ERROR: %s i=%d got=%d expected=%d\n",
                           label, (int)i, (int)values[i], (int)q);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_guard(const char *label, const uint32_t *values, uint32_t sentinel_offset) {
    int mismatches = 0;
    for (uint32_t i = WORDS; i < WORDS + GUARD_WORDS; ++i) {
        uint32_t expected = SENTINEL_BASE | (sentinel_offset + i);
        if (values[i] != expected) {
            if (mismatches < 8)
                printk("ERROR: %s guard i=%d got=%d expected=%d\n",
                       label, (int)i, (int)values[i], (int)expected);
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program) panic("vc4_program_create failed");
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(WORDS, 1, 1);
    vc4_deviceptr_t tmp_dev = 0, out_dev = 0;
    int launch_failures = 0;
    fill_buffers();
    if (vc4_m2_malloc(program, &tmp_dev, sizeof(tmp_words)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(out_words)) < 0 ||
        vc4_m2_copy_htod(program, tmp_dev, tmp_words, sizeof(tmp_words)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_words, sizeof(out_words)) < 0)
        launch_failures++;
    printk("VC4_RUNTIME_LAYOUT fixture=multi_kernel_chain program_allocations=1 code_uploads=2 heap_bytes=auto\n");
    printk("VC4_KERNEL_LAUNCH name=memory_output sequence=0\n");
    if (memory_output_launch(program, grid, block, tmp_dev) < 0) launch_failures++;
    printk("VC4_KERNEL_LAUNCH name=read_nop_write sequence=1\n");
    if (read_nop_write_launch(program, grid, block, tmp_dev, out_dev, WORDS) < 0 ||
        vc4_m2_copy_dtoh(program, tmp_words, tmp_dev, sizeof(tmp_words)) < 0 ||
        vc4_m2_copy_dtoh(program, out_words, out_dev, sizeof(out_words)) < 0)
        launch_failures++;
    int mismatches = verify_pattern("tmp", tmp_words) + verify_pattern("out", out_words);
    int sentinel_mismatches =
        verify_guard("tmp", tmp_words, 0u) + verify_guard("out", out_words, 0x100u);
    uint32_t checksum = vc4_m2_checksum_u32(out_words, WORDS);
    uint32_t runtime_allocations = memory_output_runtime_allocations();
    uint32_t runtime_launches = memory_output_runtime_launches();
    uint32_t code_uploads = memory_output_runtime_code_uploads();
    const char *status = (!mismatches && !sentinel_mismatches && !launch_failures &&
                          runtime_allocations == 1u && runtime_launches == 2u &&
                          code_uploads == 2u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=multi_kernel_chain status=%s checked_elements=%d mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d words=%d checksum=%d runtime_allocations=%d program_allocations=%d code_uploads=%d runtime_launches=%d\n",
           status, (int)(WORDS * 2u), mismatches, sentinel_mismatches, launch_failures,
           12, 16, (int)WORDS, (int)checksum, (int)runtime_allocations,
           (int)runtime_allocations, (int)code_uploads, (int)runtime_launches);
    if (status[0] != 'P')
        panic("multi_kernel_chain verification failed: mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_allocations=%d code_uploads=%d runtime_launches=%d",
              mismatches, sentinel_mismatches, launch_failures, (int)runtime_allocations,
              (int)code_uploads, (int)runtime_launches);
    vc4Free(program, tmp_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
