#include "rpi.h"
#include <stdint.h>
#include <string.h>
#include "minimal_thrend_launch.h"
#include "memory_output_launch.h"

static uint32_t out_words[MEMORY_OUTPUT_MAX_WORDS];

static uint32_t checksum_words(const uint32_t *values, uint32_t words) {
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < words; ++i) checksum += values[i];
    return checksum;
}

void notmain(void) {
    struct vc4_runtime rt;
    if (vc4_runtime_init(&rt) < 0) panic("Failed to initialize VC4 runtime");
    uint32_t active_qpus = vc4_runtime_active_qpus(&rt);
    uint32_t lanes = vc4_runtime_lane_width();
    uint32_t words = active_qpus * lanes;
    int launch_failures = 0;
    int mismatches = 0;
    memset(out_words, 0, sizeof(out_words));
    printk("Running VC4 multi_kernel_minimal reference bundle...\n");
    if (minimal_thrend_launch(&rt) < 0) launch_failures++;
    if (memory_output_launch(&rt, out_words, MEMORY_OUTPUT_MAX_WORDS) < 0) launch_failures++;
    for (uint32_t q = 0; q < active_qpus; ++q) {
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            uint32_t i = q * lanes + lane;
            if (out_words[i] != q) mismatches++;
        }
    }
    uint32_t checksum = checksum_words(out_words, words);
    const char *status = (!mismatches && !launch_failures) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=multi_kernel_minimal status=%s mismatches=%d launch_failures=%d active_qpus=%d lanes=%d words=%d checksum=%d runtime_launches=%d\n", status, mismatches, launch_failures, active_qpus, lanes, words, checksum, 2);
    vc4_runtime_shutdown(&rt);
}
