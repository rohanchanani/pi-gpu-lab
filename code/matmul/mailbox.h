#ifndef BARE_MBOX_H
#define BARE_MBOX_H

#include <stdint.h>
#include <stddef.h>

/*
 * Bare-metal Mailbox Interface for Raspberry Pi
 *
 * This header defines the functions for interacting with the GPU's mailbox
 * property interface in a bare-metal environment.
 *
 * Functions:
 *   - mailbox_write(): Write a message to the mailbox.
 *   - mailbox_read():  Read a message from the mailbox.
 *   - mailbox_call():  Send a property message and wait for the response.
 *
 *   - mem_alloc():     Allocate GPU memory.
 *   - mem_free():      Free GPU memory.
 *   - mem_lock():      Lock allocated GPU memory and return its bus address.
 *   - mem_unlock():    Unlock GPU memory.
 *
 *   - qpu_enable():    Enable (or disable) the QPU.
 *   - execute_qpu():   Execute QPU code.
 *
 * All property messages are sent on mailbox channel 8.
 *
 * Note: Ensure that the mailbox message buffers are 16-byte aligned.
 */

/* Basic mailbox I/O routines */
void mailbox_write(uint8_t channel, uint32_t data);
uint32_t mailbox_read(uint8_t channel);

/* Mailbox property call.
 * 'msg' must be 16-byte aligned.
 * Returns nonzero on success (i.e. if msg[1] == 0x80000000).
 */
int mbox_property(uint32_t *msg);

/* GPU memory allocation and management functions.
 *
 * mem_alloc:  Allocates 'size' bytes of GPU memory, with the given 'align'
 *             and 'flags'. Returns a nonzero handle on success.
 *
 * mem_free:   Releases the GPU memory associated with the given handle.
 *
 * mem_lock:   Locks the allocated GPU memory to obtain a bus address.
 *             Returns the bus address on success.
 *
 * mem_unlock: Unlocks the previously locked GPU memory.
 */
uint32_t mem_alloc(uint32_t size, uint32_t align, uint32_t flags);
uint32_t mem_free(uint32_t handle);
uint32_t mem_lock(uint32_t handle);
uint32_t mem_unlock(uint32_t handle);

/* QPU control functions.
 *
 * qpu_enable: Enable (or disable) the QPU. Pass 1 to enable, 0 to disable. */
uint32_t qpu_enable(uint32_t enable);



enum perf_counter_id {
    PERF_FEP_VALID_PRIM_NO_PIXELS = 0,
    PERF_FEP_VALID_PRIM_ALL_TILES = 1,
    PERF_FEP_CLIPPED_QUADS = 2,
    PERF_FEP_VALID_QUADS = 3,
    PERF_TLB_QUADS_NO_STENCIL = 4,
    PERF_TLB_QUADS_NO_Z_STENCIL = 5,
    PERF_TLB_QUADS_Z_STENCIL_PASS = 6,
    PERF_TLB_QUADS_ZERO_COVERAGE = 7,
    PERF_TLB_QUADS_NONZERO_COVERAGE = 8,
    PERF_TLB_QUADS_COLOR_WRITE = 9,
    PERF_PTB_PRIM_OUTSIDE_VIEWPORT = 10,
    PERF_PTB_PRIM_NEED_CLIPPING = 11,
    PERF_PSE_PRIM_REVERSED = 12,
    PERF_QPU_IDLE_CYCLES = 13,
    PERF_QPU_VERTEX_SHADING_CYCLES = 14,
    PERF_QPU_FRAGMENT_SHADING_CYCLES = 15,
    PERF_QPU_VALID_INSTR_CYCLES = 16,
    PERF_QPU_TMU_STALL_CYCLES = 17,
    PERF_QPU_SCOREBOARD_STALL_CYCLES = 18,
    PERF_QPU_VARYINGS_STALL_CYCLES = 19,
    PERF_QPU_ICACHE_HITS = 20,
    PERF_QPU_ICACHE_MISSES = 21,
    PERF_QPU_UCACHE_HITS = 22,
    PERF_QPU_UCACHE_MISSES = 23,
    PERF_TMU_TEXTURE_QUADS = 24,
    PERF_TMU_CACHE_MISSES = 25,
    PERF_VPM_VDW_STALL_CYCLES = 26,
    PERF_VPM_VCD_STALL_CYCLES = 27,
    PERF_L2C_CACHE_HITS = 28,
    PERF_L2C_CACHE_MISSES = 29
};

struct perf_ctr {
    enum perf_counter_id source_id;
    uint32_t value;
};

struct profiler_data {
    struct perf_ctr perf_ctr[16];
};

void profile_dump(struct profiler_data *profiler_data, const char *name); 

unsigned gpu_fft_base_exec_direct(uint32_t code, uint32_t unifs[], int num_qpus, struct profiler_data *profiler_data);

#endif /* BARE_MBOX_H */
