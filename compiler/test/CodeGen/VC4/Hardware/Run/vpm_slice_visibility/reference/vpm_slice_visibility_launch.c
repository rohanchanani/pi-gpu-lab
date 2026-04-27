#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "vpm_slice_visibility_launch.h"
#include "vpm_slice_visibilityshader.h"

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

#define NUM_UNIFS 9u
#define QPU_REQUEST_TIMEOUT_USEC 2000000u

#define MODE_SANITY 0u
#define MODE_VISIBILITY 1u
#define MODE_COLLISION 2u

struct vpm_slice_visibility_launch_state
{
    uint32_t code[sizeof(vpm_slice_visibilityshader) / sizeof(uint32_t)];
    uint32_t unif[2][NUM_UNIFS];
    uint32_t unif_ptr[2];
    uint32_t handle;
    struct vpm_slice_visibility_results results;
};

static uint32_t gpu_addr(const volatile void *ptr)
{
    return GPU_BASE + (uint32_t)ptr;
}

static void fill_vector(volatile uint32_t *dst, uint32_t value)
{
    for (uint32_t i = 0; i < VPM_SLICE_VISIBILITY_LANES; i++)
        dst[i] = value;
}

static uint32_t slice_of_qpu(uint32_t qpu, uint32_t qpus_per_slice)
{
    return qpus_per_slice ? (qpu / qpus_per_slice) : 0xffffffffu;
}

static void decode_ident1(volatile struct vpm_slice_visibility_results *res)
{
    uint32_t ident1 = GET32(V3D_IDENT1);
    uint32_t vpmsz_field = (ident1 >> 28) & 0xf;

    res->ident1 = ident1;
    res->vpmsz_field = vpmsz_field;
    res->vpm_kib = vpmsz_field ? vpmsz_field : 16u;
    res->tmus_per_slice = (ident1 >> 12) & 0xf;
    res->qpus_per_slice = (ident1 >> 8) & 0xf;
    res->num_slices = (ident1 >> 4) & 0xf;
    res->revision = ident1 & 0xf;
    res->num_qpus = res->qpus_per_slice * res->num_slices;
}

static void reserve_only_pair(uint32_t qpu_a, uint32_t qpu_b)
{
    uint32_t sqrsv0 = 0;
    uint32_t sqrsv1 = 0;

    for (uint32_t q = 0; q < VPM_SLICE_VISIBILITY_MAX_QPUS; q++)
    {
        uint32_t reserve_user_programs = 1u;
        if (q == qpu_a || q == qpu_b)
            reserve_user_programs = 0u;

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
        if ((uint32_t)(now - start) > QPU_REQUEST_TIMEOUT_USEC)
            return -1;
    }

    return 0;
}

static int launch_requests(
    volatile struct vpm_slice_visibility_launch_state *state,
    uint32_t request_count)
{
    if (request_count == 0 || request_count > 2)
        return -1;

    clear_scheduler_and_caches();

    for (uint32_t i = 0; i < request_count; i++)
    {
        PUT32(V3D_SRQUA, state->unif_ptr[i]);
        PUT32(V3D_SRQPC, (uint32_t)state->code);
    }

    if (wait_for_completions(request_count) < 0)
        return -1;

    state->results.srqcs_after_last_run = GET32(V3D_SRQCS);
    return 0;
}

static void set_unif_ptrs(volatile struct vpm_slice_visibility_launch_state *state)
{
    for (uint32_t i = 0; i < 2; i++)
        state->unif_ptr[i] = gpu_addr(&state->unif[i][0]);
}

static int run_sanity(
    volatile struct vpm_slice_visibility_launch_state *state)
{
    volatile struct vpm_slice_visibility_sanity_result *sanity =
        &state->results.sanity;

    fill_vector(sanity->qpu_report, VPM_SLICE_VISIBILITY_SENTINEL);
    fill_vector(sanity->observed, VPM_SLICE_VISIBILITY_SENTINEL);

    state->unif[0][0] = MODE_SANITY;
    state->unif[0][1] = gpu_addr(&sanity->qpu_report[0]);
    state->unif[0][2] = gpu_addr(&sanity->observed[0]);
    state->unif[0][3] = VPM_SLICE_VISIBILITY_TEST_ROW;

    reserve_only_pair(0, 0);
    sanity->errstat_before = GET32(V3D_ERRSTAT);
    int rc = launch_requests(state, 1);
    sanity->errstat_after = GET32(V3D_ERRSTAT);
    clear_qpu_reservations();

    if (rc < 0)
    {
        sanity->timeout = 1;
        state->results.timeouts++;
        return -1;
    }

    return 0;
}

static int run_visibility_pair(
    volatile struct vpm_slice_visibility_launch_state *state,
    uint32_t writer,
    uint32_t reader,
    uint32_t trial)
{
    volatile struct vpm_slice_visibility_results *res = &state->results;

    if (res->visibility_pair_count >= VPM_SLICE_VISIBILITY_MAX_VISIBILITY_PAIRS)
        return -1;

    uint32_t index = res->visibility_pair_count++;
    volatile struct vpm_slice_visibility_pair_result *pair =
        &res->visibility[index];

    pair->writer_qpu = writer;
    pair->reader_qpu = reader;
    pair->writer_slice = slice_of_qpu(writer, res->qpus_per_slice);
    pair->reader_slice = slice_of_qpu(reader, res->qpus_per_slice);
    pair->trial = trial;
    fill_vector(pair->observed, VPM_SLICE_VISIBILITY_SENTINEL);
    fill_vector(pair->reader_report, VPM_SLICE_VISIBILITY_SENTINEL);
    fill_vector(pair->writer_report, VPM_SLICE_VISIBILITY_SENTINEL);

    for (uint32_t i = 0; i < 2; i++)
    {
        state->unif[i][0] = MODE_VISIBILITY;
        state->unif[i][1] = writer;
        state->unif[i][2] = reader;
        state->unif[i][3] = trial;
        state->unif[i][4] = VPM_SLICE_VISIBILITY_TEST_ROW;
        state->unif[i][5] = gpu_addr(&pair->observed[0]);
        state->unif[i][6] = gpu_addr(&pair->reader_report[0]);
        state->unif[i][7] = gpu_addr(&pair->writer_report[0]);
    }

    reserve_only_pair(writer, reader);
    pair->errstat_before = GET32(V3D_ERRSTAT);
    int rc = launch_requests(state, 2);
    pair->errstat_after = GET32(V3D_ERRSTAT);
    clear_qpu_reservations();

    if (rc < 0)
    {
        pair->timeout = 1;
        res->timeouts++;
        return -1;
    }

    return 0;
}

static int run_collision_pair(
    volatile struct vpm_slice_visibility_launch_state *state,
    uint32_t qpu_a,
    uint32_t qpu_b,
    uint32_t order,
    uint32_t trial)
{
    volatile struct vpm_slice_visibility_results *res = &state->results;

    if (res->collision_run_count >= VPM_SLICE_VISIBILITY_MAX_COLLISION_RUNS)
        return -1;

    uint32_t index = res->collision_run_count++;
    volatile struct vpm_slice_visibility_collision_result *collision =
        &res->collision[index];

    collision->qpu_a = qpu_a;
    collision->qpu_b = qpu_b;
    collision->slice_a = slice_of_qpu(qpu_a, res->qpus_per_slice);
    collision->slice_b = slice_of_qpu(qpu_b, res->qpus_per_slice);
    collision->order = order;
    collision->trial = trial;
    fill_vector(collision->observed_by_a, VPM_SLICE_VISIBILITY_SENTINEL);
    fill_vector(collision->observed_by_b, VPM_SLICE_VISIBILITY_SENTINEL);
    fill_vector(collision->qpu_a_report, VPM_SLICE_VISIBILITY_SENTINEL);
    fill_vector(collision->qpu_b_report, VPM_SLICE_VISIBILITY_SENTINEL);

    for (uint32_t i = 0; i < 2; i++)
    {
        state->unif[i][0] = MODE_COLLISION;
        state->unif[i][1] = qpu_a;
        state->unif[i][2] = qpu_b;
        state->unif[i][3] = order;
        state->unif[i][4] = VPM_SLICE_VISIBILITY_TEST_ROW;
        state->unif[i][5] = gpu_addr(&collision->observed_by_a[0]);
        state->unif[i][6] = gpu_addr(&collision->observed_by_b[0]);
        state->unif[i][7] = gpu_addr(&collision->qpu_a_report[0]);
        state->unif[i][8] = gpu_addr(&collision->qpu_b_report[0]);
    }

    reserve_only_pair(qpu_a, qpu_b);
    collision->errstat_before = GET32(V3D_ERRSTAT);
    int rc = launch_requests(state, 2);
    collision->errstat_after = GET32(V3D_ERRSTAT);
    clear_qpu_reservations();

    if (rc < 0)
    {
        collision->timeout = 1;
        res->timeouts++;
        return -1;
    }

    return 0;
}

static int add_visibility_representatives(
    volatile struct vpm_slice_visibility_launch_state *state)
{
    volatile struct vpm_slice_visibility_results *res = &state->results;
    uint32_t qps = res->qpus_per_slice;
    uint32_t nslices = res->num_slices;
    uint32_t nq = res->num_qpus;
    uint32_t trial = 1;

    if (qps >= 2 && nq >= 2)
        if (run_visibility_pair(state, 0, 1, trial++) < 0)
            return -1;

    if (nslices >= 2 && qps < nq)
    {
        if (run_visibility_pair(state, 0, qps, trial++) < 0)
            return -1;
        if (run_visibility_pair(state, qps, 0, trial++) < 0)
            return -1;
    }

    if (nslices >= 3 && (2u * qps) < nq)
    {
        if (run_visibility_pair(state, 0, 2u * qps, trial++) < 0)
            return -1;
        if (run_visibility_pair(state, 2u * qps, 0, trial++) < 0)
            return -1;
    }

    return 0;
}

static int add_collision_representatives(
    volatile struct vpm_slice_visibility_launch_state *state)
{
    volatile struct vpm_slice_visibility_results *res = &state->results;
    uint32_t qps = res->qpus_per_slice;
    uint32_t nslices = res->num_slices;
    uint32_t nq = res->num_qpus;
    uint32_t trial = 20;

    if (qps >= 2 && nq >= 2)
    {
        if (run_collision_pair(state, 0, 1, 0, trial++) < 0)
            return -1;
        if (run_collision_pair(state, 0, 1, 1, trial++) < 0)
            return -1;
    }

    if (nslices >= 2 && qps < nq)
    {
        if (run_collision_pair(state, 0, qps, 0, trial++) < 0)
            return -1;
        if (run_collision_pair(state, 0, qps, 1, trial++) < 0)
            return -1;
    }

    if (nslices >= 3 && (2u * qps) < nq)
    {
        if (run_collision_pair(state, 0, 2u * qps, 0, trial++) < 0)
            return -1;
        if (run_collision_pair(state, 0, 2u * qps, 1, trial++) < 0)
            return -1;
    }

    return 0;
}

int vpm_slice_visibility_launch(
    struct vc4_runtime *rt,
    struct vpm_slice_visibility_results *out)
{
    if (!rt || !rt->isInitialized)
        return -1;
    if (!out)
        return -1;

    uint32_t handle = mem_alloc(sizeof(struct vpm_slice_visibility_launch_state),
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

    volatile struct vpm_slice_visibility_launch_state *state =
        (volatile struct vpm_slice_visibility_launch_state *)(vc - GPU_BASE);
    if (!state)
    {
        mem_unlock(handle);
        mem_free(handle);
        return -1;
    }

    memset((void *)state, 0, sizeof *state);
    state->handle = handle;
    memcpy((void *)state->code,
           vpm_slice_visibilityshader,
           sizeof state->code);
    set_unif_ptrs(state);

    decode_ident1(&state->results);

    PUT32(V3D_VPMBASE, VPM_SLICE_VISIBILITY_VPM_URSV_4K);
    state->results.vpmbase_written = VPM_SLICE_VISIBILITY_VPM_URSV_4K;
    state->results.vpmbase_readback = GET32(V3D_VPMBASE);

    clear_qpu_reservations();

    int rc = 0;
    if (state->results.num_qpus == 0 ||
        state->results.qpus_per_slice == 0 ||
        state->results.num_slices == 0 ||
        state->results.num_qpus > VPM_SLICE_VISIBILITY_MAX_QPUS)
    {
        rc = -1;
    }
    else
    {
        if (run_sanity(state) < 0)
            rc = -1;
        else if (add_visibility_representatives(state) < 0)
            rc = -1;
        else if (add_collision_representatives(state) < 0)
            rc = -1;
    }

    clear_qpu_reservations();
    memcpy(out, (const void *)&state->results, sizeof *out);

    mem_unlock(handle);
    mem_free(handle);
    return rc;
}
