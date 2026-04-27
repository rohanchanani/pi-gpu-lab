#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "qpu_barrier_syncthreads_launch.h"
#include "qpu_barrier_syncthreadsshader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define V3D_BASE 0x20C00000
#define V3D_IDENT1 (V3D_BASE + 0x00004)
#define V3D_L2CACTL (V3D_BASE + 0x00020)
#define V3D_SLCACTL (V3D_BASE + 0x00024)
#define V3D_SQRSV0 (V3D_BASE + 0x00410)
#define V3D_SQRSV1 (V3D_BASE + 0x00414)
#define V3D_SRQPC (V3D_BASE + 0x00430)
#define V3D_SRQUA (V3D_BASE + 0x00434)
#define V3D_SRQCS (V3D_BASE + 0x0043c)
#define V3D_VPMBASE (V3D_BASE + 0x00504)
#define V3D_DBCFG (V3D_BASE + 0x00e00)
#define V3D_DBQITE (V3D_BASE + 0x00e2c)
#define V3D_DBQITC (V3D_BASE + 0x00e30)
#define V3D_ERRSTAT (V3D_BASE + 0x00f20)

#define NUM_UNIFS 14u

struct qpu_barrier_launch_state
{
    uint32_t code[sizeof(qpu_barrier_syncthreadsshader) / sizeof(uint32_t)];
    uint32_t unif[QPU_BARRIER_MAX_WARPS][NUM_UNIFS];
    uint32_t unif_ptr[QPU_BARRIER_MAX_WARPS];
    uint32_t handle;
    struct qpu_barrier_results results;
};

struct qpu_barrier_run_config
{
    uint32_t mode;
    uint32_t block_count;
    uint32_t warps_per_block;
    uint32_t iterations;
    uint32_t allowed_qpu_mask;
    uint32_t vpm_rows_per_block;
};

static uint32_t gpu_addr(const volatile void *ptr)
{
    return GPU_BASE + (uint32_t)ptr;
}

static uint32_t low_mask(uint32_t n)
{
    if (n >= 32u)
        return 0xffffffffu;
    return (1u << n) - 1u;
}

static void decode_ident1(volatile struct qpu_barrier_results *res)
{
    uint32_t ident1 = GET32(V3D_IDENT1);
    uint32_t vpmsz_field = (ident1 >> 28) & 0xf;

    res->ident1 = ident1;
    res->vpmsz_field = vpmsz_field;
    res->vpm_kib = vpmsz_field ? vpmsz_field : 16u;
    res->num_semaphores = (ident1 >> 16) & 0xffu;
    res->qpus_per_slice = (ident1 >> 8) & 0xfu;
    res->num_slices = (ident1 >> 4) & 0xfu;
    res->num_qpus = res->qpus_per_slice * res->num_slices;
}

static int topology_supports_required_modes(
    volatile const struct qpu_barrier_results *res)
{
    if (res->num_qpus < QPU_BARRIER_EXPECTED_QPUS)
        return 0;
    if (res->qpus_per_slice != 4u)
        return 0;
    if (res->num_slices < 3u)
        return 0;
    if (res->num_semaphores < 16u)
        return 0;
    return 1;
}

static void reserve_qpu_mask(uint32_t allowed_mask)
{
    uint32_t sqrsv0 = 0;
    uint32_t sqrsv1 = 0;

    for (uint32_t q = 0; q < QPU_BARRIER_MAX_QPUS; q++)
    {
        uint32_t reserve_user_programs =
            ((allowed_mask & (1u << q)) == 0) ? 1u : 0u;

        if (q < 8u)
            sqrsv0 |= reserve_user_programs << (4u * q);
        else
            sqrsv1 |= reserve_user_programs << (4u * (q - 8u));
    }

    PUT32(V3D_SQRSV0, sqrsv0);
    PUT32(V3D_SQRSV1, sqrsv1);
}

static void clear_qpu_reservations(void)
{
    PUT32(V3D_SQRSV0, 0);
    PUT32(V3D_SQRSV1, 0);
}

static void clear_scheduler_and_caches(void)
{
    PUT32(V3D_DBCFG, 0);
    PUT32(V3D_DBQITE, 0);
    PUT32(V3D_DBQITC, 0xffffffffu);
    PUT32(V3D_L2CACTL, 1u << 2);
    PUT32(V3D_SLCACTL, 0xffffffffu);

    /* Clear queue, queue error, request count, and completion count. */
    PUT32(V3D_SRQCS, (1u << 0) | (1u << 7) | (1u << 8) | (1u << 16));
}

static int wait_for_completions(uint32_t expected)
{
    uint32_t start = (uint32_t)timer_get_usec();

    while ((((GET32(V3D_SRQCS) >> 16) & 0xffu) != expected))
    {
        uint32_t now = (uint32_t)timer_get_usec();
        if ((uint32_t)(now - start) > QPU_BARRIER_TIMEOUT_USEC)
            return -1;
    }

    return 0;
}

static void fill_vector(volatile uint32_t vec[QPU_BARRIER_LANES],
                        uint32_t value)
{
    for (uint32_t i = 0; i < QPU_BARRIER_LANES; i++)
        vec[i] = value;
}

static void clear_warp_result(
    volatile struct qpu_barrier_warp_result *warp,
    uint32_t block_id,
    uint32_t logical_warp_id)
{
    warp->block_id = block_id;
    warp->logical_warp_id = logical_warp_id;
    warp->reserved0 = 0;
    warp->reserved1 = 0;
    fill_vector(warp->physical_qpu_report, QPU_BARRIER_SENTINEL);
    fill_vector(warp->all_seen_mask_by_lane, QPU_BARRIER_SENTINEL);
    fill_vector(warp->always_seen_mask_by_lane, QPU_BARRIER_SENTINEL);
    fill_vector(warp->mismatch_count_by_lane, QPU_BARRIER_SENTINEL);
}

static void prepare_run_result(
    volatile struct qpu_barrier_run_result *run,
    uint32_t run_id,
    const struct qpu_barrier_run_config *cfg)
{
    memset((void *)run, 0, sizeof *run);

    run->run_id = run_id;
    run->mode = cfg->mode;
    run->block_count = cfg->block_count;
    run->warps_per_block = cfg->warps_per_block;
    run->total_requests = cfg->block_count * cfg->warps_per_block;
    run->iterations = cfg->iterations;
    run->sem_base = 0;
    run->vpm_base_row[0] = 0;
    run->vpm_base_row[1] = 16;
    run->vpm_base_row[2] = 32;
    run->vpm_base_row[3] = 48;
    run->vpm_rows_per_block = cfg->vpm_rows_per_block;
    run->qpu_set_mask = cfg->allowed_qpu_mask;
    run->expected_qpu_mask = cfg->allowed_qpu_mask;
    run->observed_qpu_mask = 0;
    run->first_bad_iter = 0xffffffffu;
    run->first_bad_block = 0xffffffffu;
    run->first_bad_reader_warp = 0xffffffffu;
    run->first_bad_peer_warp = 0xffffffffu;
    run->first_bad_lane = 0xffffffffu;
    run->first_bad_observed = QPU_BARRIER_SENTINEL;
    run->first_bad_expected = QPU_BARRIER_SENTINEL;

    for (uint32_t block = 0; block < cfg->block_count; block++)
    {
        for (uint32_t warp = 0; warp < cfg->warps_per_block; warp++)
        {
            uint32_t request = block * cfg->warps_per_block + warp;
            clear_warp_result(&run->warp[request], block, warp);
        }
    }
}

static int launch_run(
    volatile struct qpu_barrier_launch_state *state,
    uint32_t run_index,
    const struct qpu_barrier_run_config *cfg)
{
    volatile struct qpu_barrier_run_result *run =
        &state->results.runs[run_index];
    uint32_t total_requests = cfg->block_count * cfg->warps_per_block;

    if (total_requests == 0 || total_requests > QPU_BARRIER_MAX_WARPS)
        return -1;
    if (cfg->iterations == 0 || cfg->iterations > QPU_BARRIER_MAX_ITERS)
        return -1;

    prepare_run_result(run, run_index, cfg);

    for (uint32_t block = 0; block < cfg->block_count; block++)
    {
        for (uint32_t warp = 0; warp < cfg->warps_per_block; warp++)
        {
            uint32_t request = block * cfg->warps_per_block + warp;
            uint32_t sem_base = block * 4u;

            state->unif[request][0] = cfg->mode;
            state->unif[request][1] = run_index;
            state->unif[request][2] = block;
            state->unif[request][3] = warp;
            state->unif[request][4] = cfg->warps_per_block;
            state->unif[request][5] = cfg->iterations;
            state->unif[request][6] = run->vpm_base_row[block];
            state->unif[request][7] = cfg->vpm_rows_per_block;
            state->unif[request][8] = sem_base + 0u;
            state->unif[request][9] = sem_base + 1u;
            state->unif[request][10] = sem_base + 2u;
            state->unif[request][11] = sem_base + 3u;
            state->unif[request][12] = gpu_addr(&run->warp[request]);
            state->unif[request][13] = gpu_addr(run);
            state->unif_ptr[request] = gpu_addr(&state->unif[request][0]);
        }
    }

    reserve_qpu_mask(cfg->allowed_qpu_mask);
    clear_scheduler_and_caches();

    run->errstat_before = GET32(V3D_ERRSTAT);

    for (uint32_t i = 0; i < total_requests; i++)
    {
        PUT32(V3D_SRQUA, state->unif_ptr[i]);
        PUT32(V3D_SRQPC, (uint32_t)state->code);
    }

    int rc = wait_for_completions(total_requests);
    run->errstat_after = GET32(V3D_ERRSTAT);
    state->results.srqcs_after_last_run = GET32(V3D_SRQCS);
    clear_qpu_reservations();

    if (rc < 0)
    {
        run->timeout = 1;
        state->results.timeouts++;
        return -1;
    }

    return 0;
}

int qpu_barrier_syncthreads_launch(
    struct vc4_runtime *rt,
    struct qpu_barrier_results *out_results)
{
    if (!rt || !rt->isInitialized)
        return -1;
    if (!out_results)
        return -1;

    uint32_t handle = mem_alloc(sizeof(struct qpu_barrier_launch_state),
                                4096,
                                GPU_MEM_FLG);
    if (!handle)
        return -1;

    uint32_t vc = mem_lock(handle);
    if (!vc)
    {
        mem_free(handle);
        return -1;
    }

    volatile struct qpu_barrier_launch_state *state =
        (volatile struct qpu_barrier_launch_state *)(vc - GPU_BASE);
    memset((void *)state, 0, sizeof *state);
    state->handle = handle;
    memcpy((void *)state->code,
           qpu_barrier_syncthreadsshader,
           sizeof state->code);

    decode_ident1(&state->results);
    state->results.vpmbase_written = QPU_BARRIER_VPM_URSV_4K;
    PUT32(V3D_VPMBASE, QPU_BARRIER_VPM_URSV_4K);
    state->results.vpmbase_readback = GET32(V3D_VPMBASE);

    uint32_t qps = state->results.qpus_per_slice;
    uint32_t cross1 = qps;
    uint32_t cross2 = 2u * qps;

    struct qpu_barrier_run_config configs[5];
    configs[0] = (struct qpu_barrier_run_config){
        QPU_BARRIER_MODE_SAME_SLICE_SMOKE,
        1u,
        2u,
        4u,
        (1u << 0) | (1u << 1),
        16u,
    };
    configs[1] = (struct qpu_barrier_run_config){
        QPU_BARRIER_MODE_CROSS_SLICE_0_1,
        1u,
        2u,
        4u,
        (1u << 0) | (1u << cross1),
        16u,
    };
    configs[2] = (struct qpu_barrier_run_config){
        QPU_BARRIER_MODE_CROSS_SLICE_0_2,
        1u,
        2u,
        4u,
        (1u << 0) | (1u << cross2),
        16u,
    };
    configs[3] = (struct qpu_barrier_run_config){
        QPU_BARRIER_MODE_FULL_BLOCK_STRESS,
        1u,
        12u,
        64u,
        low_mask(12u),
        32u,
    };
    configs[4] = (struct qpu_barrier_run_config){
        QPU_BARRIER_MODE_TWO_BLOCK_PARTITION,
        2u,
        4u,
        32u,
        low_mask(8u),
        16u,
    };

    int rc = 0;
    state->results.run_count = 5;

    if (!topology_supports_required_modes(&state->results))
    {
        rc = -1;
    }
    else
    {
        for (uint32_t i = 0; i < state->results.run_count; i++)
        {
            if (launch_run(state, i, &configs[i]) < 0)
            {
                rc = -1;
                break;
            }
        }
    }

    memcpy(out_results, (const void *)&state->results, sizeof *out_results);

    clear_qpu_reservations();
    mem_unlock(handle);
    mem_free(handle);
    return rc;
}
