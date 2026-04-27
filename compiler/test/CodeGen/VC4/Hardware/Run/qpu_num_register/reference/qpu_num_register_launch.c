#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "qpu_num_register_launch.h"
#include "qpu_num_registershader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define NUM_UNIFS 1

/*
 * ARM-visible VC4 V3D register base.
 *
 * If rpi.h already defines one of these base names, this code uses it.  The
 * final fallback is the BCM2835/Pi0/Pi1 ARM physical V3D base.  On Pi2/Pi3, if
 * your rpi.h does not define a peripheral base, compile with:
 *
 *   -DQPU_NUM_REGISTER_V3D_BASE=0x3FC00000u
 */
#ifndef QPU_NUM_REGISTER_V3D_BASE
# ifdef V3D_BASE
#  define QPU_NUM_REGISTER_V3D_BASE V3D_BASE
# elif defined(V3D_BASE_ADDR)
#  define QPU_NUM_REGISTER_V3D_BASE V3D_BASE_ADDR
# elif defined(RPI_V3D_BASE)
#  define QPU_NUM_REGISTER_V3D_BASE RPI_V3D_BASE
# elif defined(BCM2835_V3D_BASE)
#  define QPU_NUM_REGISTER_V3D_BASE BCM2835_V3D_BASE
# elif defined(PERIPHERAL_BASE)
#  define QPU_NUM_REGISTER_V3D_BASE (PERIPHERAL_BASE + 0x00C00000u)
# elif defined(RPI_PERIPHERAL_BASE)
#  define QPU_NUM_REGISTER_V3D_BASE (RPI_PERIPHERAL_BASE + 0x00C00000u)
# elif defined(RPI_MMIO_BASE)
#  define QPU_NUM_REGISTER_V3D_BASE (RPI_MMIO_BASE + 0x00C00000u)
# elif defined(MMIO_BASE)
#  define QPU_NUM_REGISTER_V3D_BASE (MMIO_BASE + 0x00C00000u)
# else
#  define QPU_NUM_REGISTER_V3D_BASE 0x20C00000u
# endif
#endif

#define V3D_IDENT0  0x00000u
#define V3D_IDENT1  0x00004u
#define V3D_IDENT2  0x00008u
#define V3D_SQRSV0  0x00410u
#define V3D_SQRSV1  0x00414u
#define V3D_SRQCS   0x0043Cu
#define V3D_VPMBASE 0x00504u
#define V3D_DBQITE  0x00E2Cu
#define V3D_DBQITC  0x00E30u
#define V3D_ERRSTAT 0x00F20u

#define V3D_IDENT0_IDSTR_V3D 0x443356u
#define V3D_ALL_QPU_BITS     0x0000ffffu
#define V3D_VPMURSV_4KB      16u

struct qpu_num_register_launch_state
{
    uint32_t code[sizeof(qpu_num_registershader) / sizeof(uint32_t)];
    uint32_t unif[QPU_NUM_REGISTER_LAUNCHES][NUM_UNIFS];
    uint32_t unif_ptr[QPU_NUM_REGISTER_LAUNCHES];
    uint32_t handle;
    uint32_t payload[];
};

static struct qpu_num_register_debug last_debug;

static volatile uint32_t *v3d_reg(uint32_t offset)
{
    return (volatile uint32_t *)(QPU_NUM_REGISTER_V3D_BASE + offset);
}

static uint32_t v3d_read(uint32_t offset)
{
    return *v3d_reg(offset);
}

static void v3d_write(uint32_t offset, uint32_t value)
{
    *v3d_reg(offset) = value;
}

static size_t qpu_num_register_state_size(void)
{
    return offsetof(struct qpu_num_register_launch_state, payload) +
           (size_t)QPU_NUM_REGISTER_LAUNCHES *
               QPU_NUM_REGISTER_LANE_WIDTH * sizeof(uint32_t);
}

static uint32_t *qpu_num_register_out_ptr(
    volatile struct qpu_num_register_launch_state *state)
{
    return (uint32_t *)state->payload;
}

static uint32_t qpu_mask_for_count(uint32_t count)
{
    if (count >= QPU_NUM_REGISTER_MAX_QPUS)
        return V3D_ALL_QPU_BITS;
    if (count == 0)
        return 0;
    return (1u << count) - 1u;
}

static uint32_t decode_hw_qpus(struct vc4_runtime *rt, uint32_t ident0, uint32_t ident1)
{
    uint32_t qpus_per_slice = (ident1 >> 8) & 0xfu;
    uint32_t slices = (ident1 >> 4) & 0xfu;
    uint32_t hw_qpus = qpus_per_slice * slices;

    if ((ident0 & 0x00ffffffu) != V3D_IDENT0_IDSTR_V3D)
        hw_qpus = 0;

    if (hw_qpus == 0 || hw_qpus > QPU_NUM_REGISTER_MAX_QPUS)
        hw_qpus = vc4_runtime_active_qpus(rt);
    if (hw_qpus == 0 || hw_qpus > QPU_NUM_REGISTER_MAX_QPUS)
        hw_qpus = QPU_NUM_REGISTER_LAUNCHES;

    return hw_qpus;
}

static void reserve_all_except(uint32_t keep_qpu)
{
    uint32_t rsv0 = 0;
    uint32_t rsv1 = 0;

    for (uint32_t q = 0; q < QPU_NUM_REGISTER_MAX_QPUS; q++)
    {
        if (q == keep_qpu)
            continue;

        if (q < 8)
            rsv0 |= 1u << (4u * q);        /* QPURSVq bit 0: no user programs */
        else
            rsv1 |= 1u << (4u * (q - 8u)); /* QPURSVq bit 0: no user programs */
    }

    v3d_write(V3D_SQRSV0, rsv0);
    v3d_write(V3D_SQRSV1, rsv1);
}

static void fill_words(uint32_t *p, uint32_t words, uint32_t value)
{
    for (uint32_t i = 0; i < words; i++)
        p[i] = value;
}

static void capture_before(struct vc4_runtime *rt)
{
    memset(&last_debug, 0, sizeof last_debug);

    last_debug.ident0 = v3d_read(V3D_IDENT0);
    last_debug.ident1 = v3d_read(V3D_IDENT1);
    last_debug.ident2 = v3d_read(V3D_IDENT2);

    last_debug.qpus_per_slice = (last_debug.ident1 >> 8) & 0xfu;
    last_debug.num_slices = (last_debug.ident1 >> 4) & 0xfu;
    last_debug.hw_qpus = decode_hw_qpus(rt, last_debug.ident0, last_debug.ident1);
    last_debug.active_qpus = last_debug.hw_qpus;
    last_debug.expected_mask = qpu_mask_for_count(last_debug.active_qpus);

    last_debug.sqrsv0_before = v3d_read(V3D_SQRSV0);
    last_debug.sqrsv1_before = v3d_read(V3D_SQRSV1);
    last_debug.vpm_base_before = v3d_read(V3D_VPMBASE);
    last_debug.dbqite_before = v3d_read(V3D_DBQITE);
    last_debug.errstat_before = v3d_read(V3D_ERRSTAT);
}

static void restore_v3d_state(void)
{
    v3d_write(V3D_SQRSV0, last_debug.sqrsv0_before);
    v3d_write(V3D_SQRSV1, last_debug.sqrsv1_before);
    v3d_write(V3D_DBQITE, last_debug.dbqite_before);
    v3d_write(V3D_DBQITC, V3D_ALL_QPU_BITS);

    /* Preserve the runtime's original VPM user reservation. */
    v3d_write(V3D_VPMBASE, last_debug.vpm_base_before);

    last_debug.sqrsv0_after = v3d_read(V3D_SQRSV0);
    last_debug.sqrsv1_after = v3d_read(V3D_SQRSV1);
    last_debug.vpm_base_after = v3d_read(V3D_VPMBASE);
    last_debug.dbqite_after = v3d_read(V3D_DBQITE);
    last_debug.srqcs_after = v3d_read(V3D_SRQCS);
    last_debug.errstat_after = v3d_read(V3D_ERRSTAT);
}

const struct qpu_num_register_debug *qpu_num_register_last_debug(void)
{
    return &last_debug;
}

int qpu_num_register_launch(
    struct vc4_runtime *rt,
    uint32_t out[QPU_NUM_REGISTER_LAUNCHES])
{
    if (!rt || !rt->isInitialized)
        return -1;
    if (!out)
        return -1;

    capture_before(rt);

    size_t allocSize = qpu_num_register_state_size();
    uint32_t handle = mem_alloc(allocSize, 4096, GPU_MEM_FLG);
    if (!handle)
        return -1;

    uint32_t vc = mem_lock(handle);
    if (!vc)
    {
        mem_free(handle);
        return -1;
    }

    volatile struct qpu_num_register_launch_state *state =
        (volatile struct qpu_num_register_launch_state *)(vc - GPU_BASE);
    if (!state)
    {
        mem_unlock(handle);
        mem_free(handle);
        return -1;
    }

    state->handle = handle;
    memcpy((void *)state->code, qpu_num_registershader, sizeof state->code);

    uint32_t *gpuOut = qpu_num_register_out_ptr(state);
    fill_words(gpuOut,
               QPU_NUM_REGISTER_LAUNCHES * QPU_NUM_REGISTER_LANE_WIDTH,
               QPU_NUM_REGISTER_SENTINEL);

    for (uint32_t q = 0; q < QPU_NUM_REGISTER_LAUNCHES; q++)
        out[q] = QPU_NUM_REGISTER_ABSENT;

    for (uint32_t target = 0; target < QPU_NUM_REGISTER_LAUNCHES; target++)
    {
        uint32_t *slot = gpuOut + target * QPU_NUM_REGISTER_LANE_WIDTH;
        state->unif[target][0] = GPU_BASE + (uint32_t)slot;
        state->unif_ptr[target] = GPU_BASE + (uint32_t)&state->unif[target];
    }

    /* Reserve 4 KiB of user VPM and enable/clear HOST_INT latches. */
    v3d_write(V3D_VPMBASE, V3D_VPMURSV_4KB);
    v3d_write(V3D_DBQITC, V3D_ALL_QPU_BITS);
    v3d_write(V3D_DBQITE, last_debug.dbqite_before | last_debug.expected_mask);

    for (uint32_t target = 0; target < last_debug.active_qpus; target++)
    {
        uint32_t one_unif_ptr = state->unif_ptr[target];

        reserve_all_except(target);
        v3d_write(V3D_DBQITC, V3D_ALL_QPU_BITS);

        gpu_fft_base_exec_direct((uint32_t)state->code,
                                 &one_unif_ptr,
                                 1);

        uint32_t observed = gpuOut[target * QPU_NUM_REGISTER_LANE_WIDTH];
        out[target] = observed;

        if (observed < QPU_NUM_REGISTER_MAX_QPUS)
            last_debug.seen_mask |= 1u << observed;

        last_debug.irq_mask |= v3d_read(V3D_DBQITC) & V3D_ALL_QPU_BITS;
    }

    restore_v3d_state();

    mem_unlock(handle);
    mem_free(handle);
    return 0;
}
