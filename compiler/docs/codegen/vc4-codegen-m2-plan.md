# VC4 Codegen Milestone 2 Plan

**Document status:** proposed M2 implementation plan after M1 completion  
**Scope:** finish the scheduled-VC4 artifact/runtime backend so later `vc4tile -> ssavc4 -> vc4` lowering has a stable CUDA-like target
**Non-scope:** SSAVC4 lowering, register allocation, instruction scheduling from SSA ops, or `gpu` dialect lowering

---

## 1. Executive decisions

### 1.1 M2 scope boundary

M2 is the backend milestone for already-scheduled VC4 QPU kernels.

The M2 input boundary is:

```text
MLIR module containing one vc4.module
  containing one or more kernel vc4.func ops
  with domain = qpu
  with form = scheduled
  with vc4.launch_abi / resource metadata
  with bodies already expressed as vc4.qpu.* scheduled sink operations
```

The M2 output boundary is:

```text
versioned artifact bundle
  manifest.json schema v2
  layout.json
  one qasm artifact per kernel
  generated launcher/runtime C/H
  assembled code arrays for all kernels
  candidate workdir that builds and runs on Pi hardware
```

M2 should not lower pre-scheduled SSA VC4 IR to scheduled QPU instructions. That remains M3 and will use a separate `ssavc4` dialect. M2 may add metadata contracts that `ssavc4` and the later VC4 Tile dialect (`vc4tile`) will target.

After the post-M2 cleanup, active `vc4` is only the scheduled sink dialect:
TMU/SFU/VPM/VDW/DMA behavior appears as scheduled QPU bundles, load-immediates,
semaphores, branches, raw register accesses, signals, setup immediates, and
wait/synchronization sequences. The removed structured `vc4.*` op families are
not part of the active M2 sink and must not be reintroduced as an M2 shortcut.

### 1.2 CUDA-like host ABI decision

M2 should **not** generate a public host convenience wrapper that automatically allocates device memory, copies `in` buffers to the device, launches, copies `out` buffers back, and frees device memory.

Reason: this does not match normal CUDA user experience. CUDA/NVCC provides generated host-side launch stubs and runtime support for kernel configuration, argument passing, and module/function management, but user code normally handles memory allocation and host-device copies explicitly through APIs such as `cudaMalloc`, `cudaMemcpy`, and `cudaFree`. Unified memory is a separate allocation model, not a compiler-generated copy wrapper.

M2 should instead expose a CUDA-like runtime surface:

```c
typedef uint32_t vc4_deviceptr_t;

typedef struct vc4_dim3 {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} vc4_dim3;

struct vc4_program;

int vc4_program_create(struct vc4_program **out, uint32_t requested_bytes);
void vc4_program_destroy(struct vc4_program *program);

int vc4Malloc(struct vc4_program *program, vc4_deviceptr_t *out, uint32_t bytes);
int vc4Free(struct vc4_program *program, vc4_deviceptr_t ptr);
int vc4MemcpyHtoD(struct vc4_program *program, vc4_deviceptr_t dst,
                  const void *src, uint32_t bytes);
int vc4MemcpyDtoH(struct vc4_program *program, void *dst,
                  vc4_deviceptr_t src, uint32_t bytes);
int vc4MemcpyDtoD(struct vc4_program *program, vc4_deviceptr_t dst,
                  vc4_deviceptr_t src, uint32_t bytes);
int vc4MemsetD8(struct vc4_program *program, vc4_deviceptr_t dst,
                uint8_t value, uint32_t bytes);

int <kernel_public_name>_launch(struct vc4_program *program,
                                vc4_dim3 grid,
                                vc4_dim3 block,
                                /* buffer args as vc4_deviceptr_t */,
                                /* scalar args by value */);
```

The compiler-generated launch function must pack uniforms and enqueue the kernel. It must not copy host buffers.

`in`, `out`, and `inout` remain useful metadata in `vc4.launch_abi` and `manifest.json`, but in M2 they describe expected kernel memory effects, verification expectations, and future frontend lowering contracts. They do **not** instruct the generated runtime to allocate/copy/free host data.

Tests may use non-public helper functions to reduce fixture boilerplate, but these helpers must live in test support code or fixture harnesses, not in the generated public kernel ABI.

### 1.3 Scheduling placement decision

M2 should include runtime launch scheduling from scheduled VC4 kernels to hardware queues. M2 should not include high-level compiler scheduling from `gpu` dialect to VC4.

This split is precise:

| Concern | M2 backend? | M3/lowering? |
|---|---:|---:|
| Choose code descriptor for a kernel ID | yes | no |
| Pack launch uniforms for already-declared ABI entries | yes | no |
| Split total logical work into resident QPU request waves | yes | no |
| Assign logical request/warp IDs to uniforms | yes | no |
| Allocate VPM row ranges and semaphore sets for resident barrier blocks | yes | no |
| Enqueue all warps of a cooperative barrier block as a resident wave | yes | no |
| Reject resource metadata that cannot run safely | yes | partly |
| Lower `gpu.block_id` / `gpu.thread_id` / `gpu.barrier` to VC4 metadata/code | no | yes |
| Generate the four-semaphore barrier protocol from structured IR | no | yes |
| Allocate QPU registers or schedule ALU instructions | no | yes |

So M2-07 and M2-08 remain in M2, but they must be described as **runtime launch schedulers/resource allocators for already-scheduled kernels**, not as structured compiler lowering.

### 1.4 General artifact-bundle mechanism

The verifier mechanism should not be called `multi_kernel_artifact_bundle` in a way that implies a different path for single-kernel bundles. Use a general mechanism named:

```json
"mechanism": "program_artifact_bundle"
```

A one-kernel program is just `kernels.length == 1`. All M2 slices should use the same manifest, layout, assembly, and runtime contracts for single-kernel and multi-kernel programs.

---

## 2. Backend model after M2

A generated VC4 program image is one persistent device-visible allocation containing:

```text
program header / descriptors
kernel descriptor table
kernel 0 code blob
kernel 1 code blob
...
kernel N-1 code blob
kernel 0 uniform streams
kernel 0 uniform pointer array
kernel 1 uniform streams
kernel 1 uniform pointer array
...
runtime bookkeeping
heap payload region
```

The runtime creates this program image once, uploads code once, and reuses it across launches. Every launch chooses a kernel descriptor, writes uniforms into that kernel's uniform stream area, flushes or invalidates caches as required by runtime policy, enqueues SRQUA/SRQPC pairs using the resident code address, and waits/streams according to schedule mode.

M2 must support both:

```text
independent_vector schedule mode
  many independent logical QPU requests
  no barrier/shared-memory residency constraint
  waves of up to active_qpus requests

cooperative_block schedule mode
  one CUDA-like block is 1..12 logical QPU warps
  if barrier/shared VPM is used, all warps in a block must be resident together
  each resident block receives disjoint VPM rows and semaphore IDs
```

---

## 3. M2 slices

### M2-00: M2 verifier/scaffold and specification baseline

#### Expected end state

The repo has a first-class M2 descriptor, worklist, verification spec, context profiles, prompt templates, and generic milestone automation that consumes those files. M1 tests still pass.

M2 is now an example milestone consumed by the generic milestone scripts. Future milestones provide a descriptor plus worklist, verifications, context profiles, and prompts. M1-specific expected state is not a persistent acceptance target; M2 and later milestones are cumulative.

#### Source products

```text
pro_scripts/vc4_codegen_m2_worklist.json
pro_scripts/vc4_codegen_m2_verifications.json
pro_scripts/vc4_codegen_m2_context_profiles.json
pro_scripts/vc4_milestone_verifier.py
pro_scripts/vc4_milestone_autorun.py
pro_scripts/vc4_milestone_resume.py
pro_scripts/vc4_milestone_failure_router.py
pro_scripts/milestones/vc4-codegen-m2.json
pro_scripts/prompts/vc4_codegen_m2/constitution.md
pro_scripts/prompts/vc4_codegen_m2/output_contract.md
pro_scripts/prompts/vc4_codegen_m2/slice_contract.md
compiler/docs/codegen/vc4-codegen-m2-plan.md
```

#### Verifications

```json
[
  {
    "id": "m2-source-products",
    "mechanism": "source_products",
    "files": [
      "pro_scripts/vc4_codegen_m2_worklist.json",
      "pro_scripts/vc4_codegen_m2_verifications.json",
      "pro_scripts/vc4_codegen_m2_context_profiles.json"
    ]
  },
  {
    "id": "build-vc4-codegen",
    "mechanism": "build",
    "target": "vc4-codegen"
  },
  {
    "id": "build-check-vc4",
    "mechanism": "build",
    "target": "check-vc4"
  },
  {
    "id": "lit-existing-emit-suite",
    "mechanism": "lit",
    "root": "compiler/build/test/CodeGen/VC4/Emit"
  },
  {
    "id": "m1-saxpy-full-still-passes",
    "mechanism": "fixture_matrix",
    "fixtures": ["saxpy_full"],
    "phases": ["generate", "assemble", "build", "candidate_hardware", "expected_json"]
  }
]
```

#### Non-goals

No backend feature changes except scaffolding. No artifact schema changes yet.

---

### M2-01: Manifest v2 for general program artifacts

#### Expected end state

`vc4-codegen` accepts one `vc4.module` with one or more scheduled kernel functions and emits `manifest.json` schema v2 with a `kernels[]` array. There is no separate single-kernel schema. Existing single-kernel tests are migrated to schema v2; M2 verification must not require or recreate the M1 root `kernel.qasm` singleton layout.

#### Manifest v2 required shape

```json
{
  "schema_version": 2,
  "kind": "vc4-codegen-artifact-bundle",
  "program_name": "...",
  "target": {
    "name": "vc4-bcm2835-user-qpu",
    "warp_size": 16,
    "max_active_qpus": 12,
    "shared_vpm_bytes": 4096,
    "semaphores": 16
  },
  "kernels": [
    {
      "kernel_id": 0,
      "symbol_name": "mlir_symbol",
      "public_name": "c_public_name",
      "qasm_path": "kernels/c_public_name.qasm",
      "code_symbol": "c_public_name_shader",
      "scheduled_sink_ops": 123,
      "uniform_words_per_request": 6,
      "max_requests_per_wave": 12,
      "tail_policy": "tail_safe",
      "schedule_mode": "independent_vector",
      "args": [],
      "builtins": [],
      "resources": {
        "uses_barrier": false,
        "uses_shared_vpm": false,
        "vpm_bytes_per_block": 0,
        "semaphores_per_block": 0,
        "warps_per_block_max": 1
      }
    }
  ]
}
```

#### Verifications

```json
[
  {
    "id": "emit-multi-kernel-manifest-v2-lit",
    "mechanism": "lit",
    "root": "compiler/build/test/CodeGen/VC4/Emit"
  },
  {
    "id": "manifest-v2-single-kernel",
    "mechanism": "manifest_schema",
    "bundle": ".vc4_auto/codegen_m2/candidates/saxpy_full",
    "schema_version": 2,
    "expected_kernel_count": 1
  },
  {
    "id": "manifest-v2-two-kernel",
    "mechanism": "manifest_schema",
    "bundle": ".vc4_auto/codegen_m2/candidates/multi_kernel_minimal",
    "schema_version": 2,
    "expected_kernel_count": 2,
    "unique_kernel_fields": ["kernel_id", "public_name", "code_symbol", "qasm_path"]
  },
  {
    "id": "reject-duplicate-public-name",
    "mechanism": "negative_diagnostic",
    "argv": ["vc4-codegen", "compiler/test/CodeGen/VC4/Emit/emit-multi-kernel-invalid-duplicate-public-name.mlir", "--emit-bundle", "%t.bundle"],
    "stderr_contains": ["duplicate", "public_name"]
  }
]
```

#### Tests to create

```text
compiler/test/CodeGen/VC4/Emit/emit-manifest-v2-single-kernel.mlir
compiler/test/CodeGen/VC4/Emit/emit-multi-kernel-manifest-v2.mlir
compiler/test/CodeGen/VC4/Emit/emit-multi-kernel-invalid-duplicate-public-name.mlir
compiler/test/CodeGen/VC4/Emit/emit-multi-kernel-invalid-duplicate-code-symbol.mlir
```

---

### M2-02: General artifact bundle and all-kernel assembly

#### Expected end state

The support scripts and verifier treat every bundle as a program bundle. They read `manifest.json`, assemble every kernel QASM listed in `kernels[].qasm_path`, produce one code array per kernel, and copy/link all code arrays into the candidate workdir. A one-kernel bundle follows exactly the same path.

#### Required support-script behavior

`compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh` must:

1. locate `manifest.json`,
2. parse `kernels[]`,
3. assemble every `qasm_path`,
4. use each kernel's `code_symbol` for generated `.c/.h`,
5. avoid hardcoding root `kernel.qasm`; QASM comes from `manifest.kernels[].qasm_path`, and code arrays come from `manifest.kernels[].code_symbol`,
6. preserve compatibility with current fixture layout,
7. keep generated logs out of git.

#### Verifications

```json
[
  {
    "id": "program-artifact-bundle-single",
    "mechanism": "program_artifact_bundle",
    "bundle": ".vc4_auto/codegen_m2/candidates/saxpy_full",
    "expected_kernel_count": 1
  },
  {
    "id": "program-artifact-bundle-two-kernel",
    "mechanism": "program_artifact_bundle",
    "bundle": ".vc4_auto/codegen_m2/candidates/multi_kernel_minimal",
    "expected_kernel_count": 2
  },
  {
    "id": "assemble-all-qasm-multi-kernel",
    "mechanism": "all_qasm_assemble",
    "bundle": ".vc4_auto/codegen_m2/candidates/multi_kernel_minimal"
  },
  {
    "id": "candidate-assemble-multi-kernel-minimal",
    "mechanism": "candidate_phase",
    "name": "multi_kernel_minimal",
    "phase": "assemble"
  },
  {
    "id": "candidate-build-multi-kernel-minimal",
    "mechanism": "candidate_phase",
    "name": "multi_kernel_minimal",
    "phase": "build"
  },
  {
    "id": "support-script-no-single-kernel-only-assumption",
    "mechanism": "support_script_contract",
    "script": "compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh",
    "required_patterns": ["manifest.json", "kernels", "qasm_path", "code_symbol"]
  }
]
```

---

### M2-03: Persistent program image and layout.json

#### Expected end state

Generated runtime creates a single persistent device-visible program allocation per compiled program. It copies every kernel code blob once during setup and then reuses resident code addresses for all launches. Launch functions do not allocate, free, or recopy code.

#### Required layout.json shape

```json
{
  "schema_version": 1,
  "program_name": "...",
  "alignment_bytes": 8,
  "program_bytes": 1048576,
  "static_bytes": 16384,
  "heap_offset_bytes": 16384,
  "heap_size_bytes": 1032192,
  "regions": [
    {"name": "descriptor_table", "offset": 0, "size": 256, "alignment": 8},
    {"name": "kernel_0_code", "kernel_id": 0, "offset": 256, "size": 512, "alignment": 8},
    {"name": "kernel_0_uniforms", "kernel_id": 0, "offset": 768, "size": 288, "alignment": 8},
    {"name": "kernel_0_unif_ptrs", "kernel_id": 0, "offset": 1056, "size": 48, "alignment": 8}
  ],
  "kernels": [
    {
      "kernel_id": 0,
      "code_offset": 256,
      "code_bytes": 512,
      "uniform_stream_offset": 768,
      "uniform_words_per_request": 6,
      "max_requests_per_wave": 12,
      "uniform_ptr_offset": 1056,
      "uniform_ptr_count": 12
    }
  ]
}
```

#### Verifications

```json
[
  {
    "id": "layout-no-overlap-single",
    "mechanism": "program_layout_contract",
    "bundle": ".vc4_auto/codegen_m2/candidates/saxpy_full",
    "require_no_overlaps": true,
    "alignment_bytes": 8,
    "require_heap": false
  },
  {
    "id": "runtime-setup-only-code-upload",
    "mechanism": "generated_runtime_contract",
    "bundle": ".vc4_auto/codegen_m2/candidates/saxpy_full",
    "allocation_policy": {
      "program_image_allocations": "exactly_one_in_setup",
      "code_upload": "setup_only",
      "launch_reuses_resident_code": true
    },
    "forbidden_patterns_in_launch_functions": ["mem_alloc", "mem_lock", "mem_free", "copy_kernel_code"]
  },
  {
    "id": "runtime-event-log-persistent-single",
    "mechanism": "runtime_event_log",
    "fixture": "saxpy_full",
    "required_counters": {
      "program_allocations": 1,
      "code_uploads": 1
    }
  }
]
```

---

### M2-04: Device heap inside persistent allocation

#### Expected end state

The runtime exposes CUDA-like allocation/copy/free APIs backed by a heap carved from the tail of the persistent program image. User buffers are not static fields in the generated program struct. The heap is deterministic, alignment-aware, bounds-checked, and suitable for hardware tests.

#### Public runtime API

```c
int vc4Malloc(struct vc4_program *program, vc4_deviceptr_t *out, uint32_t bytes);
int vc4Free(struct vc4_program *program, vc4_deviceptr_t ptr);
int vc4MemcpyHtoD(struct vc4_program *program, vc4_deviceptr_t dst,
                  const void *src, uint32_t bytes);
int vc4MemcpyDtoH(struct vc4_program *program, void *dst,
                  vc4_deviceptr_t src, uint32_t bytes);
int vc4MemcpyDtoD(struct vc4_program *program, vc4_deviceptr_t dst,
                  vc4_deviceptr_t src, uint32_t bytes);
int vc4MemsetD8(struct vc4_program *program, vc4_deviceptr_t dst,
                uint8_t value, uint32_t bytes);
```

#### Heap minimum behavior

1. 8-byte aligned allocations by default.
2. Free-list allocator with split/free/coalesce.
3. Deterministic first-fit or best-fit policy.
4. `vc4Free(NULL-equivalent)` is either no-op or deterministic error, documented explicitly.
5. Double free rejected in debug/test mode.
6. Out-of-range device pointer rejected.
7. Copy APIs check heap bounds.
8. Program static regions are not allocatable.

#### Verifications

```json
[
  {
    "id": "heap-layout-present",
    "mechanism": "program_layout_contract",
    "bundle": ".vc4_auto/codegen_m2/candidates/heap_smoke",
    "require_heap": true,
    "min_heap_bytes": 4096
  },
  {
    "id": "heap-api-unit",
    "mechanism": "heap_api_unit",
    "source_files": [
      "compiler/test/CodeGen/VC4/Runtime/heap_unit_test.c"
    ],
    "expect_exit_code": 0,
    "stdout_contains": [
      "HEAP_UNIT_RESULT status=PASS",
      "coalesce_pass=1",
      "alignment_pass=1",
      "invalid_free_rejected=1"
    ]
  },
  {
    "id": "generated-runtime-has-vc4-memory-api",
    "mechanism": "generated_runtime_contract",
    "bundle": ".vc4_auto/codegen_m2/candidates/heap_smoke",
    "required_functions": [
      "vc4Malloc",
      "vc4Free",
      "vc4MemcpyHtoD",
      "vc4MemcpyDtoH",
      "vc4MemcpyDtoD",
      "vc4MemsetD8"
    ],
    "forbidden_patterns": ["static uint8_t .*user", "static float .*input", "static float .*output"]
  }
]
```

---

### M2-05: CUDA-like public launch ABI, no host copy wrapper

#### Expected end state

Generated launch headers expose CUDA-like kernel launch functions that accept a program handle, grid/block launch geometry, device pointers for buffer arguments, and by-value scalar arguments. The compiler-generated launch path packs uniforms and enqueues hardware. It does not allocate, copy, or free user buffers.

#### Public generated declarations

For a kernel whose `public_name` is `saxpy_full`, the header should contain a form equivalent to:

```c
int saxpy_full_launch(struct vc4_program *program,
                      vc4_dim3 grid,
                      vc4_dim3 block,
                      vc4_deviceptr_t y,
                      vc4_deviceptr_t x,
                      uint32_t n,
                      float scale);
```

The exact argument order follows `vc4.launch_abi.args` order.

#### What happens to `direction = in/out/inout`

In M2:

```text
buffer direction = in      -> manifest/runtime metadata only
buffer direction = out     -> manifest/runtime metadata only
buffer direction = inout   -> manifest/runtime metadata only
```

It must not produce implicit copies. Test harness code may use directions to build fixture-specific setup/teardown helpers, but generated public ABI must remain device-pointer based.

#### Verifications

```json
[
  {
    "id": "cuda-like-launch-abi-saxpy-full",
    "mechanism": "launch_abi_contract",
    "bundle": ".vc4_auto/codegen_m2/candidates/saxpy_full",
    "kernels": [
      {
        "public_name": "saxpy_full",
        "launch_function": "saxpy_full_launch",
        "requires_program_handle": true,
        "requires_grid_block": true,
        "buffers_are_deviceptr": true,
        "scalars_by_value": true,
        "forbid_host_pointer_launch_args": true,
        "forbid_public_uniform_arrays": true,
        "forbid_implicit_host_copies": true
      }
    ]
  },
  {
    "id": "generated-runtime-launch-no-copies",
    "mechanism": "generated_runtime_contract",
    "bundle": ".vc4_auto/codegen_m2/candidates/saxpy_full",
    "forbidden_patterns_in_launch_functions": [
      "vc4Malloc",
      "vc4Free",
      "vc4MemcpyHtoD",
      "vc4MemcpyDtoH",
      "memcpy(.*host",
      "host.*memcpy"
    ],
    "required_patterns_in_launch_functions": [
      "uniform",
      "SRQUA",
      "SRQPC"
    ]
  },
  {
    "id": "saxpy-full-device-pointer-hardware",
    "mechanism": "fixture_matrix",
    "fixtures": ["saxpy_full"],
    "phases": ["generate", "assemble", "build", "candidate_hardware", "expected_json"]
  }
]
```

#### Tests to create/adapt

```text
compiler/test/CodeGen/VC4/Emit/emit-cuda-like-launch-abi.mlir
compiler/test/CodeGen/VC4/Emit/emit-buffer-direction-metadata-no-wrapper.mlir
compiler/test/CodeGen/VC4/Runtime/launch_abi_unit_test.c
Adapt saxpy_full candidate/run.sh to allocate/copy/free explicitly using vc4Malloc/vc4Memcpy*/vc4Free.
```

---

### M2-06: First true multi-kernel hardware program

#### Expected end state

A hardware fixture contains one generated program with at least two kernels. Both kernels live in the same persistent program image. The test allocates a heap buffer once, launches kernel A, then launches kernel B using the same buffer, and checks the final result.

#### New fixture

```text
compiler/test/CodeGen/VC4/Hardware/Run/multi_kernel_chain/input.mlir
compiler/test/CodeGen/VC4/Hardware/Run/multi_kernel_chain/expected.json
compiler/test/CodeGen/VC4/Hardware/Run/multi_kernel_chain/run.sh
compiler/test/CodeGen/VC4/Hardware/Run/multi_kernel_chain/reference/...
compiler/test/CodeGen/VC4/Hardware/Run/multi_kernel_chain/candidate/run.sh
```

#### Semantic shape

```text
kernel A: out[i] = input[i] + bias
kernel B: out[i] = out[i] * scale
host oracle: final[i] = (input[i] + bias) * scale
```

#### Verifications

```json
[
  {
    "id": "multi-kernel-chain-source-products",
    "mechanism": "source_products",
    "files": [
      "compiler/test/CodeGen/VC4/Hardware/Run/multi_kernel_chain/input.mlir",
      "compiler/test/CodeGen/VC4/Hardware/Run/multi_kernel_chain/expected.json",
      "compiler/test/CodeGen/VC4/Hardware/Run/multi_kernel_chain/run.sh",
      "compiler/test/CodeGen/VC4/Hardware/Run/multi_kernel_chain/candidate/run.sh"
    ]
  },
  {
    "id": "multi-kernel-chain-manifest",
    "mechanism": "manifest_schema",
    "bundle": ".vc4_auto/codegen_m2/candidates/multi_kernel_chain",
    "schema_version": 2,
    "expected_kernel_count": 2
  },
  {
    "id": "multi-kernel-chain-layout",
    "mechanism": "program_layout_contract",
    "bundle": ".vc4_auto/codegen_m2/candidates/multi_kernel_chain",
    "require_no_overlaps": true,
    "require_heap": true
  },
  {
    "id": "multi-kernel-chain-hardware",
    "mechanism": "fixture_matrix",
    "fixtures": ["multi_kernel_chain"],
    "phases": ["reference_hardware", "generate", "assemble", "build", "candidate_hardware", "expected_json"]
  },
  {
    "id": "multi-kernel-chain-runtime-events",
    "mechanism": "runtime_event_log",
    "fixture": "multi_kernel_chain",
    "required_counters": {
      "program_allocations": 1,
      "code_uploads": 2,
      "launch_failures": 0
    },
    "min_counters": {
      "runtime_launches": 2
    }
  }
]
```

---

### M2-07: Independent-vector runtime launch scheduler

#### Expected end state

For kernels marked `schedule_mode = independent_vector`, the generated runtime splits the grid into waves of logical QPU requests. It assigns logical request IDs, block IDs, warp IDs, and tail metadata through uniforms according to the manifest ABI. It never uses physical `QPU_NUMBER` for normal work indexing.

This is backend work because the scheduled QPU kernel already exists and expects uniform-provided logical IDs. The runtime must choose how to feed those uniforms to hardware waves.

#### M2 responsibilities

1. Compute total logical requests from `grid`, `block`, `warp_size = 16`, and manifest metadata.
2. Batch requests into waves of at most `active_qpus`.
3. Write one uniform stream per logical request in the current wave.
4. Write `unif_ptr[request]` for every request.
5. Enqueue each request with the selected kernel's resident code address.
6. Wait for completion according to current synchronous fixture policy.
7. Maintain runtime counters.

#### Non-goals

1. Lowering `gpu.thread_id` to element/lane expressions.
2. Generating predicate code for tails.
3. Generating kernel bodies.

Those are M3/compiler-lowering concerns. M2 only packs declared tail/logical-index uniforms.

#### Verifications

```json
[
  {
    "id": "independent-vector-resource-contract",
    "mechanism": "resource_contract",
    "bundle": ".vc4_auto/codegen_m2/candidates/saxpy_full",
    "kernels": [
      {
        "public_name": "saxpy_full",
        "schedule_mode": "independent_vector",
        "uses_barrier": false,
        "uses_shared_vpm": false
      }
    ]
  },
  {
    "id": "independent-vector-runtime-events",
    "mechanism": "runtime_event_log",
    "fixture": "saxpy_full",
    "contains": ["VC4_KERNEL_LAUNCH", "schedule_mode=independent_vector"],
    "not_contains": ["physical_qpu_indexing=1"]
  },
  {
    "id": "independent-vector-fixture-matrix",
    "mechanism": "fixture_matrix",
    "fixtures": [
      "memory_output",
      "read_nop_write",
      "saxpy_16",
      "saxpy_basic",
      "saxpy_full",
      "global_store_coalesced_multi"
    ],
    "phases": ["generate", "assemble", "build", "candidate_hardware", "expected_json"]
  }
]
```

---

### M2-08: Cooperative-block runtime scheduler and resource allocator

#### Expected end state

For kernels marked `schedule_mode = cooperative_block`, the generated runtime schedules full resident block waves. If a block uses barrier/shared VPM, all logical warps in that block must be resident before any warp can reach the barrier. The runtime allocates disjoint VPM rows and semaphore IDs per resident block and reclaims them only after the whole wave completes.

This is backend work because the scheduled QPU program already contains the barrier/VPM/semaphore protocol, but the runtime must provide safe residency and resource assignment. The lowering that creates the barrier protocol is M3.

#### M2 responsibilities

1. Read kernel resource metadata from manifest.
2. Reject impossible launches before hardware enqueue.
3. Compute `warps_per_block` from launch geometry or explicit manifest metadata.
4. Ensure `warps_per_block <= active_qpus` for barrier kernels.
5. Compute resident blocks per wave from QPU, VPM, and semaphore constraints.
6. Allocate VPM row ranges per resident block.
7. Allocate semaphore ID ranges per resident block.
8. Pack per-warp uniforms with logical block ID, logical warp ID, VPM base row, semaphore base, and tail metadata.
9. Enqueue all warps in all resident blocks for the wave.
10. Wait for wave completion before reusing VPM/semaphore resources.

#### Verifications

```json
[
  {
    "id": "cooperative-resource-contract-qpu-barrier",
    "mechanism": "resource_contract",
    "bundle": ".vc4_auto/codegen_m2/candidates/qpu_barrier_syncthreads",
    "target": {
      "active_qpus": 12,
      "warp_size": 16,
      "vpm_bytes": 4096,
      "hardware_semaphores": 16
    },
    "kernels": [
      {
        "public_name": "qpu_barrier_syncthreads",
        "schedule_mode": "cooperative_block",
        "uses_barrier": true,
        "require_full_block_residency": true,
        "warps_per_block_max": 12,
        "semaphores_per_block": 4
      }
    ],
    "expect": "accept"
  },
  {
    "id": "cooperative-invalid-too-many-warps",
    "mechanism": "negative_diagnostic",
    "argv": ["vc4-codegen", "compiler/test/CodeGen/VC4/Emit/emit-resource-invalid-too-many-warps.mlir", "--emit-bundle", "%t.bundle"],
    "stderr_contains": ["warps_per_block", "12"]
  },
  {
    "id": "cooperative-fixture-matrix",
    "mechanism": "fixture_matrix",
    "fixtures": [
      "qpu_barrier_syncthreads",
      "vpm_slice_visibility"
    ],
    "phases": ["generate", "assemble", "build", "candidate_hardware", "expected_json"]
  },
  {
    "id": "cooperative-runtime-event-log",
    "mechanism": "runtime_event_log",
    "fixture": "qpu_barrier_syncthreads",
    "contains": [
      "schedule_mode=cooperative_block",
      "resident_blocks=",
      "semaphore_base=",
      "vpm_base_row="
    ],
    "not_contains": ["oversubscribed_barrier=1", "early_resource_reuse=1"]
  }
]
```

---

### M2-09: Memory subsystem coverage matrix

#### Expected end state

The backend passes a curated matrix of scheduled-VC4 hardware fixtures exercising TMU loads, SFU reads, VPM/VDW stores, reductions, and tail behavior. Fixtures that require structured lowering or physical-QPU hardware characterization are explicitly deferred and not counted as M2 failures.

#### Matrix groups

```text
Smoke/runtime:
  minimal_thrend
  memory_output
  read_nop_write
  saxpy_full
  multi_kernel_chain

Independent-vector:
  saxpy_16
  saxpy_basic
  saxpy_full
  global_store_coalesced_multi
  gemv_naive_tail

Memory/SFU paths:
  saxpy_tmu
  saxpy_tmu_overlap
  tmu_read_nop_write
  tmu_strided_load
  sfu_recip
  vpm_slice_visibility

Cooperative/shared/reduction:
  qpu_barrier_syncthreads
  warp_reduce_sum
  warp_prefix_sum
  block_reduce_sum
  shared_transpose_16x16

Post-M2 hardware characterization:
  vpm_setup_clobber

Realistic/stress, stretch if scheduled inputs are already M2-compatible:
  matmul_naive
  matmul_blocked
  conv1d_3tap
  image_sobel_3x3
  image_boxblur_shared
  stencil2d_5point_naive
  stencil2d_5point_shared
  layernorm_row
  softmax_row
```

#### Verifications

```json
[
  {
    "id": "m2-smoke-runtime-matrix",
    "mechanism": "fixture_matrix",
    "matrix": "smoke_runtime",
    "required": true
  },
  {
    "id": "m2-independent-vector-matrix",
    "mechanism": "fixture_matrix",
    "matrix": "independent_vector",
    "required": true
  },
  {
    "id": "m2-memory-sfu-matrix",
    "mechanism": "fixture_matrix",
    "matrix": "memory_sfu",
    "required": true
  },
  {
    "id": "m2-cooperative-matrix",
    "mechanism": "fixture_matrix",
    "matrix": "cooperative_shared_reduction",
    "required": true
  }
]
```

---

### M2-10: Final M2 acceptance and unsupported-feature diagnostics

#### Expected end state

Every fixture in the M2 required matrix either passes candidate hardware verification or is intentionally outside M2 and has a deterministic unsupported diagnostic. No hidden single-kernel assumptions, static user buffers, per-launch code uploads, or public host-copy wrappers remain.

#### Final acceptance verifications

```json
[
  {"id": "build-vc4-codegen", "mechanism": "build", "target": "vc4-codegen"},
  {"id": "build-check-vc4", "mechanism": "build", "target": "check-vc4"},
  {"id": "lit-codegen-vc4-emit", "mechanism": "lit", "root": "compiler/build/test/CodeGen/VC4/Emit"},
  {"id": "all-required-fixtures", "mechanism": "fixture_matrix", "matrix": "m2_required", "required": true},
  {"id": "all-manifests-schema-valid", "mechanism": "manifest_schema", "matrix": "m2_required"},
  {"id": "all-layouts-valid", "mechanism": "program_layout_contract", "matrix": "m2_required"},
  {"id": "all-runtime-contracts", "mechanism": "generated_runtime_contract", "matrix": "m2_required"},
  {"id": "all-runtime-event-logs", "mechanism": "runtime_event_log", "matrix": "m2_required"},
  {"id": "reference-bundles-clean", "mechanism": "reference_immutable", "matrix": "m2_required"},
  {"id": "no-generated-artifacts-committed", "mechanism": "forbidden_absent", "globs": ["compiler/test/**/Output/**", "compiler/test/**/.lit_test_times.txt", "compiler/test/**/run.log"]}
]
```

#### Unsupported diagnostics required before M3

The backend must deterministically reject:

```text
atomics
unsupported global scatter stores
barrier kernels with warps_per_block > active_qpus
barrier kernels whose shared/VPM rows exceed the 4 KiB window
cooperative kernels requiring more semaphores than available
unsupported private spills
unsupported dynamic shared memory if metadata is missing
public host-pointer kernel launches
raw public uniform-array launch APIs
multiple vc4.modules in one artifact bundle input
empty program with no kernels
```

---

## 4. New verifier mechanisms

### 4.1 `manifest_schema`

#### Purpose

Validate `manifest.json` schema v2 for any program bundle.

#### Contract

```json
{
  "mechanism": "manifest_schema",
  "bundle": ".vc4_auto/codegen_m2/candidates/saxpy_full",
  "manifest": "manifest.json",
  "schema_version": 2,
  "expected_kernel_count": 1,
  "min_kernel_count": 1,
  "max_kernel_count": null,
  "required_top_level_keys": ["schema_version", "kind", "program_name", "target", "kernels"],
  "required_kernel_keys": [
    "kernel_id",
    "symbol_name",
    "public_name",
    "qasm_path",
    "code_symbol",
    "scheduled_sink_ops",
    "uniform_words_per_request",
    "max_requests_per_wave",
    "tail_policy",
    "schedule_mode",
    "args",
    "builtins",
    "resources"
  ],
  "unique_kernel_fields": ["kernel_id", "public_name", "code_symbol", "qasm_path"],
  "forbid_top_level_keys": ["kernel"],
  "allow_legacy_compat_keys": false
}
```

#### Consumption

Open `bundle/manifest.json`, validate schema version, required keys, kernel count, uniqueness, path safety, and optional exact values.

---

### 4.2 `program_artifact_bundle`

#### Purpose

Validate filesystem shape for any program bundle, single-kernel or multi-kernel.

#### Contract

```json
{
  "mechanism": "program_artifact_bundle",
  "bundle": ".vc4_auto/codegen_m2/candidates/multi_kernel_chain",
  "manifest": "manifest.json",
  "layout": "layout.json",
  "expected_kernel_count": 2,
  "required_common_files": ["kernel_launch.c", "kernel_launch.h", "manifest.json", "layout.json"],
  "qasm_path_source": "manifest.kernels[].qasm_path",
  "require_qasm_files": true,
  "require_unique_qasm_files": true,
  "not_contains_in_qasm": ["TODO", "placeholder"],
  "paths_must_be_bundle_relative": true
}
```

#### Consumption

Read manifest, check required files, check all QASM paths exist under the bundle, scan for forbidden text, and check `layout.json` exists when required.

---

### 4.3 `all_qasm_assemble`

#### Purpose

Assemble every kernel QASM artifact in the manifest.

#### Contract

```json
{
  "mechanism": "all_qasm_assemble",
  "bundle": ".vc4_auto/codegen_m2/candidates/multi_kernel_chain",
  "manifest": "manifest.json",
  "out_dir": ".vc4_auto/codegen_m2/candidates/multi_kernel_chain/assembled",
  "symbol_field": "code_symbol",
  "qasm_field": "qasm_path",
  "require_outputs": [".c", ".h"]
}
```

#### Consumption

For each manifest kernel, run vc4asm on its `qasm_path`, generate `<code_symbol>.c/.h`, and verify outputs exist.

---

### 4.4 `program_layout_contract`

#### Purpose

Validate `layout.json` and static region arithmetic.

#### Contract

```json
{
  "mechanism": "program_layout_contract",
  "bundle": ".vc4_auto/codegen_m2/candidates/saxpy_full",
  "layout": "layout.json",
  "manifest": "manifest.json",
  "alignment_bytes": 8,
  "require_no_overlaps": true,
  "require_heap": true,
  "min_heap_bytes": 4096,
  "max_requests_per_wave": 12,
  "expected_kernel_count": 1
}
```

#### Consumption

Check offsets, sizes, alignment, no-overlap, heap placement, heap size, kernel count, uniform stream dimensions, and uniform pointer counts.

---

### 4.5 `generated_runtime_contract`

#### Purpose

Validate generated `kernel_launch.c/.h` against backend policy.

#### Contract

```json
{
  "mechanism": "generated_runtime_contract",
  "bundle": ".vc4_auto/codegen_m2/candidates/saxpy_full",
  "header": "kernel_launch.h",
  "source": "kernel_launch.c",
  "required_functions": ["vc4_program_create", "vc4_program_destroy", "vc4Malloc", "vc4Free"],
  "required_patterns": ["struct vc4_kernel_desc", "code_gpu_addr", "unif_ptr"],
  "forbidden_patterns": ["TODO", "placeholder"],
  "forbidden_patterns_in_launch_functions": ["vc4Malloc", "vc4Free", "vc4MemcpyHtoD", "vc4MemcpyDtoH", "mem_alloc", "mem_free"],
  "allocation_policy": {
    "program_image_allocations": "exactly_one_in_setup",
    "code_upload": "setup_only",
    "launch_reuses_resident_code": true
  }
}
```

#### Consumption

Regex scan initially; optionally graduate to a C parser later. The verifier must identify generated launch functions by manifest `public_name` and inspect only those functions for stricter launch rules.

---

### 4.6 `heap_api_unit`

#### Purpose

Run host-only heap allocator tests.

#### Contract

```json
{
  "mechanism": "heap_api_unit",
  "source_files": ["compiler/test/CodeGen/VC4/Runtime/heap_unit_test.c"],
  "include_dirs": ["compiler/test/CodeGen/VC4/Runtime"],
  "stub_files": ["compiler/test/CodeGen/VC4/Runtime/vc4_mailbox_host_stub.c"],
  "expect_exit_code": 0,
  "stdout_contains": ["HEAP_UNIT_RESULT status=PASS"]
}
```

#### Consumption

Compile locally with host compiler, run locally, check stdout.

---

### 4.7 `launch_abi_contract`

#### Purpose

Validate CUDA-like public launch ABI and no implicit host copies.

#### Contract

```json
{
  "mechanism": "launch_abi_contract",
  "bundle": ".vc4_auto/codegen_m2/candidates/saxpy_full",
  "manifest": "manifest.json",
  "header": "kernel_launch.h",
  "source": "kernel_launch.c",
  "kernels": [
    {
      "public_name": "saxpy_full",
      "launch_function": "saxpy_full_launch",
      "requires_program_handle": true,
      "requires_grid_block": true,
      "buffers_are_deviceptr": true,
      "scalars_by_value": true,
      "forbid_host_pointer_launch_args": true,
      "forbid_public_uniform_arrays": true,
      "forbid_implicit_host_copies": true
    }
  ]
}
```

#### Consumption

Cross-check manifest args with header declarations and source behavior. Reject launch signatures containing host pointer buffer args like `float *x` unless they are explicitly `vc4_deviceptr_t` or equivalent device-pointer typedef.

---

### 4.8 `runtime_event_log`

#### Purpose

Parse stable runtime diagnostics from hardware logs.

#### Contract

```json
{
  "mechanism": "runtime_event_log",
  "fixture": "multi_kernel_chain",
  "log_path": "compiler/test/CodeGen/VC4/Hardware/Run/multi_kernel_chain/candidate/run.log",
  "required_counters": {
    "program_allocations": 1,
    "launch_failures": 0,
    "heap_alloc_failures": 0
  },
  "min_counters": {
    "runtime_launches": 2
  },
  "contains": ["VC4_RUNTIME_LAYOUT", "VC4_KERNEL_LAUNCH", "VC4_HEAP_STATS"],
  "not_contains": ["ERROR:", "per_launch_code_upload=1"]
}
```

#### Required log lines

Generated/test runtime should print stable lines such as:

```text
VC4_RUNTIME_LAYOUT program_bytes=<n> static_bytes=<n> heap_offset=<n> heap_bytes=<n> kernels=<n>
VC4_KERNEL_LAUNCH kernel_id=<k> public_name=<name> schedule_mode=<mode> requests=<n> waves=<n>
VC4_HEAP_STATS allocs=<n> frees=<n> failures=<n> high_water=<n>
```

---

### 4.9 `fixture_matrix`

#### Purpose

Declaratively run groups of hardware fixtures through reference, generate, assemble, build, candidate, and expected-json phases.

#### Contract

```json
{
  "mechanism": "fixture_matrix",
  "fixtures": [
    {
      "name": "saxpy_full",
      "input": "compiler/test/CodeGen/VC4/Hardware/Run/saxpy_full/input.mlir",
      "expected": "compiler/test/CodeGen/VC4/Hardware/Run/saxpy_full/expected.json",
      "required": true,
      "run_reference": true,
      "run_candidate": true
    }
  ],
  "phases": ["reference_hardware", "generate", "assemble", "build", "candidate_hardware", "expected_json"],
  "requires_hardware": true,
  "keep_going": true
}
```

#### Consumption

Expand into existing mechanisms and summarize per-fixture failures.

---

### 4.10 `resource_contract`

#### Purpose

Validate resource metadata and scheduler constraints.

#### Contract

```json
{
  "mechanism": "resource_contract",
  "bundle": ".vc4_auto/codegen_m2/candidates/qpu_barrier_syncthreads",
  "manifest": "manifest.json",
  "target": {
    "active_qpus": 12,
    "warp_size": 16,
    "vpm_bytes": 4096,
    "hardware_semaphores": 16
  },
  "kernels": [
    {
      "public_name": "qpu_barrier_syncthreads",
      "schedule_mode": "cooperative_block",
      "uses_barrier": true,
      "warps_per_block_max": 12,
      "semaphores_per_block": 4,
      "require_full_block_residency": true,
      "max_resident_blocks": 4
    }
  ],
  "expect": "accept"
}
```

#### Consumption

For `expect = accept`, verify constraints fit target resources. For `expect = reject`, run codegen or schema validation and check `diagnostic_contains`.

---

### 4.11 `support_script_contract`

#### Purpose

Prevent support scripts from regressing to single-kernel assumptions.

#### Contract

```json
{
  "mechanism": "support_script_contract",
  "script": "compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh",
  "required_patterns": ["manifest.json", "kernels", "qasm_path", "code_symbol"],
  "forbidden_literals": ["kernel.qasm only", "kernelshader.c only"],
  "run_smoke_fixture": "multi_kernel_minimal"
}
```

---

### 4.12 `negative_diagnostic`

#### Purpose

Run a command expected to fail and check deterministic diagnostics.

#### Contract

```json
{
  "mechanism": "negative_diagnostic",
  "argv": ["vc4-codegen", "path/to/input.mlir", "--emit-bundle", "%t.bundle"],
  "expect_exit_code": "nonzero",
  "stderr_contains": ["expected substring"],
  "stdout_contains": [],
  "forbid_bundle_created": true
}
```

---

## 5. Test plan

### 5.1 New local Emit tests

```text
compiler/test/CodeGen/VC4/Emit/emit-manifest-v2-single-kernel.mlir
  Tests: schema v2 single-kernel program shape.
  Source: adapt minimal_thrend/saxpy_full style scheduled kernel.

compiler/test/CodeGen/VC4/Emit/emit-multi-kernel-manifest-v2.mlir
  Tests: two kernels, two qasm paths, unique kernel IDs/public names/code symbols.
  Source: duplicate minimal scheduled body with distinct launch_abi.

compiler/test/CodeGen/VC4/Emit/emit-multi-kernel-invalid-duplicate-public-name.mlir
  Tests: deterministic rejection of duplicate public_name.

compiler/test/CodeGen/VC4/Emit/emit-program-layout-json.mlir
  Tests: layout.json exists, offsets present, heap optional/required by slice.

compiler/test/CodeGen/VC4/Emit/emit-cuda-like-launch-abi.mlir
  Tests: launch function takes program handle, grid/block, device ptr args, scalars.

compiler/test/CodeGen/VC4/Emit/emit-buffer-direction-metadata-no-wrapper.mlir
  Tests: direction metadata appears in manifest but does not generate host-copy wrapper.

compiler/test/CodeGen/VC4/Emit/emit-resource-independent-vector.mlir
  Tests: independent_vector resource metadata.

compiler/test/CodeGen/VC4/Emit/emit-resource-cooperative-block.mlir
  Tests: cooperative_block resource metadata.

compiler/test/CodeGen/VC4/Emit/emit-resource-invalid-too-many-warps.mlir
  Tests: reject cooperative block requiring > 12 warps.

compiler/test/CodeGen/VC4/Emit/emit-resource-invalid-too-much-shared-vpm.mlir
  Tests: reject shared/VPM requirement > 4096 bytes.
```

### 5.2 Host-only runtime tests

```text
compiler/test/CodeGen/VC4/Runtime/heap_unit_test.c
  Tests: split, free, coalesce, alignment, invalid free, double free, OOM.

compiler/test/CodeGen/VC4/Runtime/layout_unit_test.c
  Tests: generated layout constants match layout.json for a tiny generated bundle.

compiler/test/CodeGen/VC4/Runtime/launch_pack_unit_test.c
  Tests: uniforms are packed in manifest order and kernel_id selects correct descriptor.

compiler/test/CodeGen/VC4/Runtime/no_host_copy_launch_unit_test.c
  Tests: generated launch function does not call copy helpers.

compiler/test/CodeGen/VC4/Runtime/multi_kernel_stub_launch_unit_test.c
  Tests: alternating launches select distinct resident code addresses and reuse heap.

compiler/test/CodeGen/VC4/Runtime/vc4_mailbox_host_stub.c
  Provides host-side stubs for mem_alloc/mem_lock/cache ops/SRQ enqueue recording.
```

### 5.3 New hardware fixtures

```text
multi_kernel_chain
  Purpose: first true multi-kernel program image, shared heap buffer, launch A then B.
  Reference: hand-written reference bundle using existing saxpy_full/memory_output runtime patterns.
  Tests: multi-code array, multi-uniform stream, persistent heap, launch ordering.

multi_kernel_uniform_shape
  Purpose: kernels with different uniform counts.
  Reference: tiny arithmetic kernels with different scalar/buffer ABI shapes.
  Tests: non-rectangular uniform layout or explicit padded-stride layout.

heap_reuse_saxpy
  Purpose: allocate/copy/launch/free/reallocate/copy/launch/free.
  Reference: saxpy_full oracle.
  Tests: heap reuse and no static user buffers.

persistent_relaunch
  Purpose: same kernel launched many times with different scalar values.
  Reference: memory_output or saxpy_full style.
  Tests: uniform overwrite, uniform-cache policy, code_uploads=1.

cooperative_resource_partition
  Purpose: smaller derivative of qpu_barrier_syncthreads with disjoint VPM/semaphore resources.
  Reference: qpu_barrier_syncthreads protocol.
  Tests: resident-block resource partitioning.
```

### 5.4 Existing fixtures to adapt first

```text
saxpy_full
  Convert candidate runtime to vc4Malloc/vc4Memcpy*/vc4Free and device-pointer launch.
  Keep expected.json unchanged unless runtime counters intentionally change.

memory_output, read_nop_write
  First heap-backed output smoke tests.

saxpy_16, saxpy_basic
  Independent-vector launch/tail coverage.

global_store_coalesced_multi
  VPM/VDW output correctness under runtime policy.

saxpy_tmu, saxpy_tmu_overlap, tmu_read_nop_write, tmu_strided_load
  TMU direct load path coverage.

sfu_recip
  SFU issue/read coverage.

vpm_slice_visibility
  VPM visibility coverage required for M2.

vpm_setup_clobber
  Post-M2 physical-QPU/VPM setup litmus for validating whether conservative VPM setup serialization can be relaxed.

qpu_barrier_syncthreads
  Cooperative-block scheduler/resource allocator anchor.

warp_reduce_sum, warp_prefix_sum, block_reduce_sum
  Reduction and cooperative/shared behavior after qpu_barrier_syncthreads is stable.
```

---

## 6. Concrete startup deliverables for M2

Before starting implementation slices, produce these files:

```text
pro_scripts/vc4_codegen_m2_worklist.json
pro_scripts/vc4_codegen_m2_verifications.json
pro_scripts/vc4_codegen_m2_context_profiles.json
pro_scripts/vc4_milestone_verifier.py
pro_scripts/vc4_milestone_autorun.py
pro_scripts/vc4_milestone_resume.py
pro_scripts/vc4_milestone_failure_router.py
pro_scripts/milestones/vc4-codegen-m2.json
pro_scripts/prompts/vc4_codegen_m2/constitution.md
pro_scripts/prompts/vc4_codegen_m2/output_contract.md
pro_scripts/prompts/vc4_codegen_m2/slice_contract.md
compiler/docs/codegen/vc4-artifact-manifest-v2.md
compiler/docs/codegen/vc4-runtime-abi.md
compiler/docs/codegen/vc4-m2-test-matrix.md
compiler/test/CodeGen/VC4/Runtime/heap_unit_test.c
compiler/test/CodeGen/VC4/Runtime/vc4_mailbox_host_stub.c
compiler/test/CodeGen/VC4/Emit/emit-manifest-v2-single-kernel.mlir
compiler/test/CodeGen/VC4/Emit/emit-multi-kernel-manifest-v2.mlir
compiler/test/CodeGen/VC4/Emit/emit-cuda-like-launch-abi.mlir
```

The M2 verifier should initially support these mechanisms:

```text
source_products
build
lit
vc4_codegen_generate
program_artifact_bundle
manifest_schema
all_qasm_assemble
generated_runtime_contract
program_layout_contract
heap_api_unit
launch_abi_contract
runtime_event_log
fixture_matrix
resource_contract
support_script_contract
negative_diagnostic
reference_immutable
expected_json_result
forbidden_absent
```

Existing M1 mechanisms can be reused where behavior is identical.

---

## 7. Initial M2 command sequence

```bash
cd ~/Downloads/pi-gpu-lab
export PATH="$PWD/compiler/build/bin:$PATH"
export VC4_HW_ATTEMPT_TIMEOUT_SEC=60
export VC4_HARDWARE_TIMEOUT_SEC=60
export VC4_RUN_SH_MAX_ATTEMPTS=3

# Baseline before M2 changes.
git status --short --untracked-files=all
ninja -C compiler/build vc4-codegen check-vc4
llvm-lit -v compiler/build/test/CodeGen/VC4/Emit

# After adding M2 scaffold.
python3 pro_scripts/vc4_milestone_verifier.py verify \
  --repo "$PWD" \
  --milestone-config pro_scripts/milestones/vc4-codegen-m2.json \
  --slice m2-00-scaffold \
  --out .vc4_auto/codegen_m2/manual/m2-00-scaffold.json \
  --timeout-sec 7200 \
  --keep-going

# After each implementation slice.
python3 pro_scripts/vc4_milestone_verifier.py verify \
  --repo "$PWD" \
  --milestone-config pro_scripts/milestones/vc4-codegen-m2.json \
  --slice <slice-id> \
  --out .vc4_auto/codegen_m2/manual/<slice-id>.json \
  --timeout-sec 7200 \
  --keep-going

# Cleanup before commits.
rm -f compiler/test/CodeGen/VC4/Hardware/Run/*/candidate/run.log
rm -f compiler/test/CodeGen/VC4/Hardware/Run/*/reference/run.log
git status --short --untracked-files=all
```

---

## 8. Done criteria for M2

M2 is complete when:

1. `vc4-codegen` emits schema v2 program bundles for one or more kernels.
2. Single-kernel and multi-kernel bundles use the same artifact path and verifier mechanisms.
3. Every kernel has its own QASM, code array, descriptor, uniform stream region, and uniform pointer region.
4. Runtime creates one persistent program allocation and heap region.
5. Public memory API exposes `vc4Malloc`, `vc4Free`, and copy/memset functions.
6. Public launch ABI accepts program handle, grid/block launch geometry, device pointers, and scalars.
7. Generated public launch functions pack uniforms and enqueue hardware but do not copy host buffers.
8. Independent-vector scheduled kernels run through backend wave scheduling.
9. Cooperative-block scheduled kernels run through backend full-residency/resource scheduling.
10. Required fixture matrices pass candidate hardware or are explicitly unsupported with deterministic diagnostics.
11. No reference bundles, expected JSON files, generated lit Output files, run logs, or `.vc4_auto/**` artifacts are committed.
12. The backend target is stable enough for M3 `ssavc4` lowering and later `vc4tile -> ssavc4 -> vc4` lowering.


### Canonical single-kernel bundle policy

A single scheduled kernel is still a program bundle with `kernels.length == 1`. M2 source-product and lit tests must assert absence of root `kernel.qasm` and must read QASM through `manifest.kernels[].qasm_path`.
