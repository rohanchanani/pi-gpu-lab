// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage hardware-run metadata for shared_transpose_16x16.
//
// The trusted reference qasm computes a fixed 16x16 u32 transpose using one
// cooperative 4-warp block. Each logical warp stages four input rows into VPM,
// the block synchronizes with the locked four-semaphore reusable barrier, and
// each logical warp reads four output rows through vertical 32-bit VPM reads.
// Candidate/codegen execution remains disabled until the backend can emit the
// qasm and launcher bundle.

module attributes {
"vc4.hardware_run_test.name" = "shared_transpose_16x16",
"vc4.hardware_run_test.kind" = "hardware-run-reference",
"vc4.hardware_run_test.reference_kernel" = "reference/shared_transpose_16x16.qasm",
"vc4.hardware_run_test.launch_abi" = {
public_name = "shared_transpose_16x16_launch",
tail_policy = "fixed_16x16",
uniform_words_per_qpu = 7 : i32,
args = [
{name = "input", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
{name = "output", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32}
],
builtins = [
{name = "logical_warp_id", kind = "warp_id", materialization = "uniform_suffix", uniform_index = 2 : i32},
{name = "warps_per_block", kind = "warps_per_block", materialization = "uniform_suffix", uniform_index = 3 : i32},
{name = "vpm_base_row", kind = "workgroup_memory_base", materialization = "uniform_suffix", uniform_index = 4 : i32},
{name = "qpu_id", kind = "qpu_num", materialization = "uniform_suffix", uniform_index = 5 : i32},
{name = "num_qpus", kind = "num_qpus", materialization = "uniform_suffix", uniform_index = 6 : i32}
],
work_distribution = {
device_model = "one_cuda_like_sm",
block_shape = "4_qpu_warps_x_16_lanes",
tile_shape = "16_rows_x_16_columns",
lane = "ELEMENT_NUMBER",
local_rows_per_warp = 4 : i32,
row = "logical_warp_id * 4 + local_row",
output_row = "logical_warp_id * 4 + local_col",
shared_memory = "VPM rows vpm_base_row..vpm_base_row+15",
barrier = "four_semaphore_reusable_block0_semaphores_0_1_2_3"
},
memory_paths = {
global_loads = "TMU0 direct memory lookup",
workgroup_writes = "VPM horizontal 32-bit row writes",
workgroup_reads = "VPM vertical 32-bit column reads",
global_stores = "VPM staging plus VDW DMA store",
vpm_vdw_serialization = "global_mutex"
},
runtime_resource_policy = {
gpu_allocations_per_prepare = 1 : i32,
code_copies_per_prepare = 1 : i32,
semantic_launches_per_harness = 4 : i32,
qpu_requests_per_launch = 4 : i32,
vpm_base_row = 0 : i32,
semaphore_ids = "0,1,2,3"
},
test_cases = [
{name = "linear", expression = "row * 16 + col"},
{name = "tagged_row_col", expression = "0x70000000 | (row << 8) | col"},
{name = "affine_small", expression = "row * 37 + col * 11 + 5"},
{name = "diagonal_antidiagonal", expression = "diagonal and deterministic background"}
]
}
} {
}

