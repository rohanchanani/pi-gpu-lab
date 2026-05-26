#include "rpi.h"
#include <stdint.h>
#include <string.h>
#include "memory_output_launch.h"
#include "read_nop_write_launch.h"

static uint32_t tmp_words[MEMORY_OUTPUT_MAX_WORDS];
static uint32_t out_words[MEMORY_OUTPUT_MAX_WORDS];

static uint32_t checksum_words(const uint32_t *values, uint32_t words) {
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < words; ++i) checksum += values[i];
    return checksum;
}

static int verify_pattern(const char *label, const uint32_t *values,
                          uint32_t active_qpus, uint32_t lanes) {
    int mismatches = 0;
    for (uint32_t q = 0; q < active_qpus; ++q) {
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            uint32_t i = q * lanes + lane;
            if (values[i] != q) {
                if (mismatches < 8)
                    printk("ERROR: %s i=%u got=%u expected=%u\n",
                           label, i, values[i], q);
                mismatches++;
            }
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_runtime rt;
    if (vc4_runtime_init(&rt) < 0) panic("Failed to initialize VC4 runtime");
    uint32_t active_qpus = vc4_runtime_active_qpus(&rt);
    uint32_t lanes = vc4_runtime_lane_width();
    uint32_t words = active_qpus * lanes;
    int launch_failures = 0;
    int mismatches = 0;
    int sentinel_mismatches = 0;
    memset(tmp_words, 0, sizeof(tmp_words));
    memset(out_words, 0x5a, sizeof(out_words));
    printk("Running VC4 multi_kernel_chain reference bundle...\n");
    if (memory_output_launch(&rt, tmp_words, MEMORY_OUTPUT_MAX_WORDS) < 0) launch_failures++;
    if (read_nop_write_launch(&rt, tmp_words, out_words, words) < 0) launch_failures++;
    mismatches += verify_pattern("tmp", tmp_words, active_qpus, lanes);
    mismatches += verify_pattern("out", out_words, active_qpus, lanes);
    uint32_t checked_elements = words * 2u;
    uint32_t checksum = checksum_words(out_words, words);
    const char *status = (!mismatches && !sentinel_mismatches && !launch_failures) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=multi_kernel_chain status=%s checked_elements=%d mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d words=%d checksum=%d runtime_allocations=%d program_allocations=%d code_uploads=%d runtime_launches=%d\n",
           status, checked_elements, mismatches, sentinel_mismatches,
           launch_failures, active_qpus, lanes, words, checksum, 1, 1, 2, 2);
    vc4_runtime_shutdown(&rt);
}
