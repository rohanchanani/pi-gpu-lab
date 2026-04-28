#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "vpm_setup_clobber_launch.h"
#include "shader.h"

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

#define VPM_SETUP_CLOBBER_NUM_UNIFS 9u

struct vpm_setup_clobber_gpu_state {
uint32_t code[sizeof(shader) / sizeof(uint32_t)];
uint32_t unif[2][VPM_SETUP_CLOBBER_NUM_UNIFS];
uint32_t unif_ptr[2];
uint32_t handle;

uint32_t row_a[VPM_SETUP_CLOBBER_LANES];
uint32_t row_b[VPM_SETUP_CLOBBER_LANES];
uint32_t qpu_a_report[VPM_SETUP_CLOBBER_LANES];
uint32_t qpu_b_report[VPM_SETUP_CLOBBER_LANES];

};

static uint32_t gpu_addr(const volatile void *ptr)
{
return GPU_BASE + (uint32_t)ptr;
}

static uint32_t slice_of_qpu(uint32_t qpu, uint32_t qpus_per_slice)
{
return qpus_per_slice ? (qpu / qpus_per_slice) : 0xffffffffu;
}

static uint32_t expected_tag(uint32_t qpu_a, uint32_t lane)
{
return 0xe5000000u | ((qpu_a & 0x0fu) << 8) | (lane & 0x0fu);
}

static void fill_vector(volatile uint32_t vec[VPM_SETUP_CLOBBER_LANES],
uint32_t value)
{
for (uint32_t i = 0; i < VPM_SETUP_CLOBBER_LANES; i++)
vec[i] = value;
}

static void copy_vector(uint32_t dst[VPM_SETUP_CLOBBER_LANES],
volatile const uint32_t src[VPM_SETUP_CLOBBER_LANES])
{
for (uint32_t i = 0; i < VPM_SETUP_CLOBBER_LANES; i++)
dst[i] = src[i];
}

static uint32_t vector_matches_tag(
const uint32_t vec[VPM_SETUP_CLOBBER_LANES],
uint32_t qpu_a)
{
for (uint32_t lane = 0; lane < VPM_SETUP_CLOBBER_LANES; lane++) {
if (vec[lane] != expected_tag(qpu_a, lane))
return 0;
}
return 1;
}

static uint32_t qpu_report_mismatches(
const uint32_t vec[VPM_SETUP_CLOBBER_LANES],
uint32_t expected_qpu)
{
uint32_t mismatches = 0;
for (uint32_t lane = 0; lane < VPM_SETUP_CLOBBER_LANES; lane++) {
if (vec[lane] != expected_qpu)
mismatches++;
}
return mismatches;
}

static uint32_t classify_record(
struct vpm_setup_clobber_record *record)
{
record->row_a_match = vector_matches_tag(record->row_a, record->qpu_a);
record->row_b_match = vector_matches_tag(record->row_b, record->qpu_a);

if (record->row_a_match && !record->row_b_match)
    return VPM_SETUP_CLOBBER_CLASS_NO_CLOBBER;
if (!record->row_a_match && record->row_b_match)
    return VPM_SETUP_CLOBBER_CLASS_SETUP_CLOBBER;
if (record->row_a_match && record->row_b_match)
    return VPM_SETUP_CLOBBER_CLASS_BOTH;
return VPM_SETUP_CLOBBER_CLASS_OTHER;

}

const char *vpm_setup_clobber_classification_name(uint32_t classification)
{
switch (classification) {
case VPM_SETUP_CLOBBER_CLASS_NO_CLOBBER:
return "no_clobber";
case VPM_SETUP_CLOBBER_CLASS_SETUP_CLOBBER:
return "setup_clobber";
case VPM_SETUP_CLOBBER_CLASS_BOTH:
return "both";
case VPM_SETUP_CLOBBER_CLASS_OTHER:
return "other";
default:
return "unknown";
}
}

static void decode_ident1(struct vpm_setup_clobber_topology *topology)
{
uint32_t ident1 = GET32(V3D_IDENT1);
uint32_t vpmsz_field = (ident1 >> 28) & 0xfu;

topology->ident1 = ident1;
topology->vpmsz_field = vpmsz_field;
topology->vpm_kib = vpmsz_field ? vpmsz_field : 16u;
topology->num_semaphores = (ident1 >> 16) & 0xffu;
topology->qpus_per_slice = (ident1 >> 8) & 0xfu;
topology->num_slices = (ident1 >> 4) & 0xfu;
topology->num_qpus = topology->qpus_per_slice * topology->num_slices;

}

static void reserve_only_pair(uint32_t qpu_a, uint32_t qpu_b)
{
uint32_t sqrsv0 = 0;
uint32_t sqrsv1 = 0;

for (uint32_t q = 0; q < VPM_SETUP_CLOBBER_MAX_QPUS; q++) {
    uint32_t reserve_user_programs =
        (q == qpu_a || q == qpu_b) ? 0u : 1u;

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

while ((((GET32(V3D_SRQCS) >> 16) & 0xffu) != expected)) {
    uint32_t now = (uint32_t)timer_get_usec();
    if ((uint32_t)(now - start) > VPM_SETUP_CLOBBER_TIMEOUT_USEC)
        return -1;
}

return 0;

}

int vpm_setup_clobber_prepare(
struct vc4_runtime *rt,
struct vpm_setup_clobber_state *state)
{
if (!rt || !state)
return -1;

memset(state, 0, sizeof *state);

if (!rt->isInitialized) {
    if (vc4_runtime_init(rt) < 0)
        return -1;
}

uint32_t handle = mem_alloc(sizeof(struct vpm_setup_clobber_gpu_state),
                            4096,
                            GPU_MEM_FLG);
if (!handle)
    return -1;

uint32_t vc = mem_lock(handle);
if (!vc) {
    mem_free(handle);
    return -1;
}

volatile struct vpm_setup_clobber_gpu_state *gpu =
    (volatile struct vpm_setup_clobber_gpu_state *)(vc - GPU_BASE);

memset((void *)gpu, 0, sizeof *gpu);
gpu->handle = handle;

memcpy((void *)gpu->code, shader, sizeof gpu->code);
for (uint32_t i = 0; i < 2; i++)
    gpu->unif_ptr[i] = gpu_addr(&gpu->unif[i][0]);

decode_ident1(&state->topology);
state->topology.vpmbase_written = VPM_SETUP_CLOBBER_VPM_URSV_4K;
PUT32(V3D_VPMBASE, VPM_SETUP_CLOBBER_VPM_URSV_4K);
state->topology.vpmbase_readback = GET32(V3D_VPMBASE);

state->runtime = rt;
state->opaque = (void *)gpu;
state->prepared = 1;
state->runtime_allocations = 1;
state->runtime_launches = 0;
state->srqcs_after_last_run = 0;

return 0;

}

int vpm_setup_clobber_run_pair(
struct vpm_setup_clobber_state *state,
uint32_t qpu_a,
uint32_t qpu_b,
uint32_t trial,
struct vpm_setup_clobber_record *out)
{
if (!state || !state->prepared || !state->opaque || !out)
return -1;
if (qpu_a == qpu_b)
return -1;
if (qpu_a >= state->topology.num_qpus || qpu_b >= state->topology.num_qpus)
return -1;
if (qpu_a >= VPM_SETUP_CLOBBER_MAX_QPUS ||
qpu_b >= VPM_SETUP_CLOBBER_MAX_QPUS)
return -1;

volatile struct vpm_setup_clobber_gpu_state *gpu =
    (volatile struct vpm_setup_clobber_gpu_state *)state->opaque;

memset(out, 0, sizeof *out);
out->trial = trial;
out->qpu_a = qpu_a;
out->qpu_b = qpu_b;
out->slice_a = slice_of_qpu(qpu_a, state->topology.qpus_per_slice);
out->slice_b = slice_of_qpu(qpu_b, state->topology.qpus_per_slice);

fill_vector(gpu->row_a, VPM_SETUP_CLOBBER_SENTINEL);
fill_vector(gpu->row_b, VPM_SETUP_CLOBBER_SENTINEL);
fill_vector(gpu->qpu_a_report, VPM_SETUP_CLOBBER_SENTINEL);
fill_vector(gpu->qpu_b_report, VPM_SETUP_CLOBBER_SENTINEL);

for (uint32_t request = 0; request < 2; request++) {
    gpu->unif[request][0] = qpu_a;
    gpu->unif[request][1] = qpu_b;
    gpu->unif[request][2] = trial;
    gpu->unif[request][3] = VPM_SETUP_CLOBBER_CLOBBER_ROWA;
    gpu->unif[request][4] = VPM_SETUP_CLOBBER_CLOBBER_ROWB;
    gpu->unif[request][5] = gpu_addr(&gpu->row_a[0]);
    gpu->unif[request][6] = gpu_addr(&gpu->row_b[0]);
    gpu->unif[request][7] = gpu_addr(&gpu->qpu_a_report[0]);
    gpu->unif[request][8] = gpu_addr(&gpu->qpu_b_report[0]);
    gpu->unif_ptr[request] = gpu_addr(&gpu->unif[request][0]);
}

reserve_only_pair(qpu_a, qpu_b);
clear_scheduler_and_caches();

out->errstat_before = GET32(V3D_ERRSTAT);

for (uint32_t request = 0; request < 2; request++) {
    PUT32(V3D_SRQUA, gpu->unif_ptr[request]);
    PUT32(V3D_SRQPC, (uint32_t)gpu->code);
}

int rc = wait_for_completions(2u);
out->errstat_after = GET32(V3D_ERRSTAT);
state->srqcs_after_last_run = GET32(V3D_SRQCS);
clear_qpu_reservations();

if (rc < 0)
    out->timeout = 1;

copy_vector(out->row_a, gpu->row_a);
copy_vector(out->row_b, gpu->row_b);
copy_vector(out->qpu_a_report, gpu->qpu_a_report);
copy_vector(out->qpu_b_report, gpu->qpu_b_report);

out->qpu_mismatches =
    qpu_report_mismatches(out->qpu_a_report, qpu_a) +
    qpu_report_mismatches(out->qpu_b_report, qpu_b);
out->classification = classify_record(out);

state->runtime_launches++;

return rc < 0 ? -1 : 0;

}

void vpm_setup_clobber_shutdown(struct vpm_setup_clobber_state *state)
{
if (!state || !state->prepared || !state->opaque)
return;

volatile struct vpm_setup_clobber_gpu_state *gpu =
    (volatile struct vpm_setup_clobber_gpu_state *)state->opaque;
uint32_t handle = gpu->handle;

mem_unlock(handle);
mem_free(handle);

state->opaque = 0;
state->prepared = 0;

}

