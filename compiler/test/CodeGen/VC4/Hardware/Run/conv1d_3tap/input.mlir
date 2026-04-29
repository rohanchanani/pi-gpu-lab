// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage hardware-run metadata for conv1d_3tap.
//
// The trusted reference qasm computes a tail-safe clamp-to-edge 1D three-tap
// convolution over float buffers:
//
// out[i] = c0 * x[max(i - 1, 0)] + c1 * x[i] + c2 * x[min(i + 1, n - 1)]
//
// Candidate/codegen execution remains disabled until the backend can emit the
// qasm and launcher bundle.

module attributes {
"vc4.hardware_run_test.name" = "conv1d_3tap",
"vc4.hardware_run_test.kind" = "hardware-run-reference",
"vc4.hardware_run_test.reference_kernel" = "reference/conv1d_3tap.qasm",
"vc4.hardware_run_test.launch_abi" = {
public_name = "conv1d_3tap_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 8 : i32,
args = [
{name = "x", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
{name = "out", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 1 : i32},
{name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
{name = "c0", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 3 : i32},
{name = "c1", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 4 : i32},
{name = "c2", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 5 : i32}
],
builtins = [
{name = "qpu_id", kind = "qpu_num", materialization = "uniform_suffix", uniform_index = 6 : i32},
{name = "num_qpus", kind = "num_qpus", materialization = "uniform_suffix", uniform_index = 7 : i32}
],
work_distribution = {
logical_warp = "one QPU request",
lane = "ELEMENT_NUMBER",
base_element = "qpu_id * 16",
stride_elements = "num_qpus * 16",
store_tail = "dynamic_vdw_depth"
},
boundary_policy = {
left = "max(i - 1, 0)",
right = "min(i + 1, n - 1)",
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
semantic_launches_per_harness = 13 : i32,
qpu_requests_per_launch = 12 : i32
},
test_cases = [
0 : i32,
1 : i32,
2 : i32,
15 : i32,
16 : i32,
17 : i32,
31 : i32,
32 : i32,
33 : i32,
191 : i32,
192 : i32,
193 : i32,
511 : i32
]
}
} {
}

