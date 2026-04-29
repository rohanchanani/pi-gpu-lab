// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage hardware-run metadata for stencil2d_5point_naive.
//
// The trusted reference qasm computes a tail-safe clamp-to-edge 2D five-point
// stencil over float buffers:
//
// out[y, x] = center_weight * in[y, x] +
// neighbor_weight * (in[y-1, x] + in[y+1, x] +
// in[y, x-1] + in[y, x+1])
//
// with neighbor coordinates clamped to the valid image edge.
// Candidate/codegen execution remains disabled until the backend can emit the
// qasm and launcher bundle.

module attributes {
"vc4.hardware_run_test.name" = "stencil2d_5point_naive",
"vc4.hardware_run_test.kind" = "hardware-run-reference",
"vc4.hardware_run_test.reference_kernel" = "reference/stencil2d_5point_naive.qasm",
"vc4.hardware_run_test.launch_abi" = {
public_name = "stencil2d_5point_naive_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 9 : i32,
args = [
{name = "input", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
{name = "output", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 1 : i32},
{name = "width", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
{name = "height", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
{name = "total_elements", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32},
{name = "center_weight", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 5 : i32},
{name = "neighbor_weight", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 6 : i32}
],
builtins = [
{name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 7 : i32},
{name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 8 : i32}
],
work_distribution = {
logical_warp = "one QPU request",
lane = "ELEMENT_NUMBER",
row = "qpu_id + t * num_qpus",
column = "column_tile_base + lane",
private_scratch_row_stride = 32 : i32,
store_tail = "dynamic_vdw_depth"
},
boundary_policy = {
up = "max(y - 1, 0)",
down = "min(y + 1, height - 1)",
left = "max(x - 1, 0)",
right = "min(x + 1, width - 1)",
implemented_in = "qasm"
},
memory_paths = {
global_loads = "TMU0 direct memory lookup",
global_stores = "VPM staging plus VDW DMA store",
vpm_vdw_serialization = "global_mutex"
},
runtime_resource_policy = {
gpu_allocations_per_prepare = 1 : i32,
code_copies_per_prepare = 1 : i32,
semantic_launches_per_harness = 9 : i32,
qpu_requests_per_nonzero_launch = 12 : i32,
zero_sized_launches_dispatch_qpus = false
},
test_cases = [
{width = 0 : i32, height = 7 : i32},
{width = 1 : i32, height = 1 : i32},
{width = 2 : i32, height = 3 : i32},
{width = 5 : i32, height = 7 : i32},
{width = 7 : i32, height = 5 : i32},
{width = 16 : i32, height = 4 : i32},
{width = 17 : i32, height = 9 : i32},
{width = 31 : i32, height = 19 : i32},
{width = 32 : i32, height = 32 : i32}
]
}
} {
}

