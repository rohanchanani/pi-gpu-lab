#ifndef VC4_M2_CANDIDATE_TEST_HELPERS_H
#define VC4_M2_CANDIDATE_TEST_HELPERS_H

#include "rpi.h"
#include "kernel_launch.h"
#include <stdint.h>
#include <string.h>

static vc4_dim3 vc4_m2_dim3(uint32_t x, uint32_t y, uint32_t z) {
    vc4_dim3 d;
    d.x = x;
    d.y = y;
    d.z = z;
    return d;
}

static int vc4_m2_check_rc(const char *label, int rc) {
    if (rc < 0) {
        printk("ERROR: %s failed rc=%d\n", label, rc);
        return -1;
    }
    return 0;
}

static int vc4_m2_malloc(struct vc4_program *program, vc4_deviceptr_t *ptr, uint32_t bytes) {
    int rc = vc4Malloc(program, ptr, bytes);
    if (rc < 0)
        printk("ERROR: vc4Malloc bytes=%u failed rc=%d\n", bytes, rc);
    return rc;
}

static int vc4_m2_copy_htod(struct vc4_program *program, vc4_deviceptr_t dst, const void *src, uint32_t bytes) {
    int rc = vc4MemcpyHtoD(program, dst, src, bytes);
    if (rc < 0)
        printk("ERROR: vc4MemcpyHtoD bytes=%u failed rc=%d\n", bytes, rc);
    return rc;
}

static int vc4_m2_copy_dtoh(struct vc4_program *program, void *dst, vc4_deviceptr_t src, uint32_t bytes) {
    int rc = vc4MemcpyDtoH(program, dst, src, bytes);
    if (rc < 0)
        printk("ERROR: vc4MemcpyDtoH bytes=%u failed rc=%d\n", bytes, rc);
    return rc;
}

static uint32_t vc4_m2_checksum_u32(const uint32_t *values, uint32_t words) {
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < words; ++i)
        checksum += values[i];
    return checksum;
}

#endif
