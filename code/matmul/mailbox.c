#include <stdint.h>
#include "rpi.h"
#include "mailbox.h"

#define MAILBOX_BASE 0x2000B880
#define MAILBOX_READ (*(volatile uint32_t *)(MAILBOX_BASE + 0x0))
#define MAILBOX_STATUS (*(volatile uint32_t *)(MAILBOX_BASE + 0x18))
#define MAILBOX_WRITE (*(volatile uint32_t *)(MAILBOX_BASE + 0x20))

#define MAILBOX_FULL 0x80000000
#define MAILBOX_EMPTY 0x40000000

#define V3D_BASE 0x20C00000
#define V3D_SRQSC (V3D_BASE + 0x418)
#define V3D_L2CACTL (V3D_BASE + 0x020)
#define V3D_SLCACTL (V3D_BASE + 0x024)
#define V3D_SRQPC (V3D_BASE + 0x0430)
#define V3D_SRQUA (V3D_BASE + 0x0434)
#define V3D_SRQCS (V3D_BASE + 0x043c)
#define V3D_DBCFG (V3D_BASE + 0x0e00)
#define V3D_DBQITE (V3D_BASE + 0x0e2c)
#define V3D_DBQITC (V3D_BASE + 0x0e30)

// ─── Performance-counter registers ────────────────────────────────────────────
#define V3D_PCTRC   (V3D_BASE + 0x0670)   // clear bits
#define V3D_PCTRE   (V3D_BASE + 0x0674)   // enable bits
#define V3D_PCTR0   (V3D_BASE + 0x0680)   // counter 0 value
#define V3D_PCTRS0  (V3D_BASE + 0x0684)   // counter 0 source select
#define V3D_PCTR1   (V3D_BASE + 0x0688)   // counter 1 value
#define V3D_PCTRS1  (V3D_BASE + 0x068c)   // counter 1 source select

//
// Basic mailbox I/O routines
//

// Write 'data' to the mailbox on the specified channel.
void mailbox_write(uint8_t channel, uint32_t data)
{
	// Wait until mailbox is not full.
	while (MAILBOX_STATUS & MAILBOX_FULL)
	{
	}
	// Lower 4 bits are used for channel.
	MAILBOX_WRITE = (data & ~0xF) | (channel & 0xF);
}

// Read from the mailbox on the specified channel.
uint32_t mailbox_read(uint8_t channel)
{
	uint32_t data;
	while (1)
	{
		data = MAILBOX_READ;
		if ((data & 0xF) == channel)
			return data & ~0xF;
	}
}

// Perform a mailbox property call.
// The message buffer 'msg' must be 16-byte aligned.
// Returns nonzero if the call succeeded.
int mbox_property(uint32_t *msg)
{
	// Check alignment.
	if ((uint32_t)msg & 0xF) return 0;
	mailbox_write(8, (uint32_t)msg);
	while (mailbox_read(8) != (uint32_t)msg);

	return (msg[1] == 0x80000000);
}

//
// Mailbox property calls for GPU memory and QPU control.
// All property messages are sent on mailbox channel 8.
//

uint32_t mem_alloc(uint32_t size, uint32_t align, uint32_t flags)
{
	uint32_t p[9] __attribute__((aligned(16))) =
		{
			9 * sizeof(uint32_t), // size
			0x00000000,			  // process request
			0x3000c,			  // (the tag id)
			3 * sizeof(uint32_t), // (size of the buffer)
			3 * sizeof(uint32_t), // (size of the data)
			size,				  // (num bytes)
			align,				  // (alignment)
			flags,				  // (MEM_FLAG_L1_NONALLOCATING)
			0					  // end tag
		};
	assert(mbox_property(p));
	return p[5];
}

uint32_t mem_free(uint32_t handle)
{
	uint32_t p[7] __attribute__((aligned(16))) =
		{
			7 * sizeof(uint32_t), // size
			0x00000000,			  // process request
			0x3000f,			  // (the tag id)
			1 * sizeof(uint32_t), // (size of the buffer)
			1 * sizeof(uint32_t), // (size of the data)
			handle,				  // (handle)
			0					  // end tag
		};
	assert(mbox_property(p));
	return p[5];
}

uint32_t mem_lock(uint32_t handle)
{
	uint32_t p[7] __attribute__((aligned(16))) =
		{
			7 * sizeof(uint32_t), // size
			0x00000000,			  // process request
			0x3000d,			  // (the tag id)
			1 * sizeof(uint32_t), // (size of the buffer)
			1 * sizeof(uint32_t), // (size of the data)
			handle,				  // (handle)
			0					  // end tag
		};
	assert(mbox_property(p));
	return p[5];
}

uint32_t mem_unlock(uint32_t handle)
{
	uint32_t p[7] __attribute__((aligned(16))) =
		{
			7 * sizeof(uint32_t), // size
			0x00000000,			  // process request
			0x3000e,			  // (the tag id)
			1 * sizeof(uint32_t), // (size of the buffer)
			1 * sizeof(uint32_t), // (size of the data)
			handle,				  // (handle)
			0					  // end tag
		};
	assert(mbox_property(p));
	return p[5];
}

uint32_t qpu_enable(uint32_t enable)
{
	uint32_t p[7] __attribute__((aligned(16))) =
		{
			7 * sizeof(uint32_t), // size
			0x00000000,			  // process request
			0x30012,			  // (the tag id)
			1 * sizeof(uint32_t), // (size of the buffer)
			1 * sizeof(uint32_t), // (size of the data)
			enable,				  // (enable QPU)
			0					  // end tag
		};
	assert(mbox_property(p));
	return p[5];
}

void profile_dump(struct profiler_data *profiler_data, const char *name)
{
	printk("=================================================================\n");
	printk("GPU Profiler Data: %s\n", name);
	printk("=================================================================\n");
    for (int i = 0; i < 16; i++) {
		if (profiler_data->perf_ctr[i].source_id == PERF_FEP_VALID_PRIM_NO_PIXELS) {
			printk("FEP Valid primitives (no rendered pixels): %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_FEP_VALID_PRIM_ALL_TILES) {
			printk("FEP Valid primitives (all tiles): %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_FEP_CLIPPED_QUADS) {
			printk("FEP Early-Z/Near/Far clipped quads: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_FEP_VALID_QUADS) {
			printk("FEP Valid quads: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_TLB_QUADS_NO_STENCIL) {
			printk("TLB Quads (no stencil test): %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_TLB_QUADS_NO_Z_STENCIL) {
			printk("TLB Quads (no Z/stencil test): %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_TLB_QUADS_Z_STENCIL_PASS) {
			printk("TLB Quads (Z/stencil test pass): %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_TLB_QUADS_ZERO_COVERAGE) {
			printk("TLB Quads (zero coverage): %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_TLB_QUADS_NONZERO_COVERAGE) {
			printk("TLB Quads (non-zero coverage): %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_TLB_QUADS_COLOR_WRITE) {
			printk("TLB Quads (written to color buffer): %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_PTB_PRIM_OUTSIDE_VIEWPORT) {
			printk("PTB Primitives (outside viewport): %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_PTB_PRIM_NEED_CLIPPING) {
			printk("PTB Primitives (need clipping): %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_PSE_PRIM_REVERSED) {
			printk("PSE Primitives (reversed): %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_QPU_IDLE_CYCLES) {
			printk("QPU Total idle clock cycles: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_QPU_VERTEX_SHADING_CYCLES) {
			printk("QPU Total vertex/coordinate shading cycles: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_QPU_FRAGMENT_SHADING_CYCLES) {
			printk("QPU Total fragment shading cycles: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_QPU_VALID_INSTR_CYCLES) {
			printk("QPU Total valid instruction cycles: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_QPU_TMU_STALL_CYCLES) {
			printk("QPU Total TMU stall cycles: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_QPU_SCOREBOARD_STALL_CYCLES) {
			printk("QPU Total Scoreboard stall cycles: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_QPU_VARYINGS_STALL_CYCLES) {
			printk("QPU Total Varyings stall cycles: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_QPU_ICACHE_HITS) {
			printk("QPU Total instruction cache hits: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_QPU_ICACHE_MISSES) {
			printk("QPU Total instruction cache misses: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_QPU_UCACHE_HITS) {
			printk("QPU Total uniforms cache hits: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_QPU_UCACHE_MISSES) {
			printk("QPU Total uniforms cache misses: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_TMU_TEXTURE_QUADS) {
			printk("TMU Total texture quads processed: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_TMU_CACHE_MISSES) {
			printk("TMU Total texture cache misses: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_VPM_VDW_STALL_CYCLES) {
			printk("VPM VDW stall cycles: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_VPM_VCD_STALL_CYCLES) {
			printk("VPM VCD stall cycles: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_L2C_CACHE_HITS) {
			printk("L2C Total cache hits: %d\n", profiler_data->perf_ctr[i].value);
		} else if (profiler_data->perf_ctr[i].source_id == PERF_L2C_CACHE_MISSES) {
			printk("L2C Total cache misses: %d\n", profiler_data->perf_ctr[i].value);
		}
    }
	printk("=================================================================\n");
}

unsigned gpu_fft_base_exec_direct(uint32_t code,
                                  uint32_t unifs[],
                                  int       num_qpus,
                                  struct profiler_data *profiler_data)
{


    PUT32(V3D_DBCFG, 0);        // disallow IRQs
    PUT32(V3D_DBQITE, 0);
    PUT32(V3D_DBQITC, -1);

    PUT32(V3D_L2CACTL, 1 << 2); // clear L2
    PUT32(V3D_SLCACTL, -1);     // clear other caches
    PUT32(V3D_SRQCS, (1 << 7) | (1 << 8) | (1 << 16));  // reset counts/err
	if (profiler_data != NULL) {
		PUT32(V3D_PCTRC, 0xffff);
		for (int i = 0; i < 16; i++) {
			PUT32(V3D_PCTRS0 + (i * 8), profiler_data->perf_ctr[i].source_id);
		}
		PUT32(V3D_PCTRE, (1 << 31) | 0xffff); 
	}
	
	for (unsigned q = 0; q < (unsigned)num_qpus; q++) {
		PUT32(V3D_SRQUA, (uint32_t)unifs[q]);  // uniforms addr
		PUT32(V3D_SRQPC, (uint32_t)code);      // shader entry
	}
    while (((GET32(V3D_SRQCS) >> 16) & 0xFF) != (uint32_t)num_qpus) {
        /* busy-wait until all QPUs done */
    }

    if (profiler_data != NULL) {
        for (int i = 0; i < 16; i++) {
            profiler_data->perf_ctr[i].value = GET32(V3D_PCTR0 + (i * 8));
        }
    }
    return 0;
}
