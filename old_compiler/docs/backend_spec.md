## 1. Title

# VC4 MLIR Backend Interface and Artifact Specification

Historical/obsolete planning note: this document records an old MLIR `gpu`-centered backend plan. The active project stack has since changed: M4 is the VC4 Tile dialect (`vc4tile`) and `vc4tile -> ssavc4` lowering; producer adapters from Triton/IREE/MLIR-GPU-like forms into `vc4tile` are future work after M4.

## 2. Status / purpose of document

This document is the standing architecture and interface specification for the Raspberry Pi VideoCore IV backend below the MLIR `gpu` dialect. It freezes the canonical backend input interface, the generated artifact interface, and the runtime split that future implementation work must follow. It is intended to be durable ground truth for both human implementation and future LLM-guided work. fileciteturn22file0

This document is normative. Where it states that something must, must not, or remains internal, that requirement is part of the project contract unless explicitly moved into the deferred questions section.

## 3. Problem statement

The project goal is not merely to emit some Pi GPU code for one handwritten example. The goal is to build a backend for the MLIR `gpu` dialect that preserves an inspectable lowering path from higher-level compiler representations down to concrete Raspberry Pi VC4 artifacts, while ultimately producing outputs that can build and run on the Pi.

The current handwritten SAXPY reference already contains the essential pieces, but those pieces are mixed together. The current tree combines a sample host program, a device artifact, kernel-specific launcher logic, generic runtime logic, and build-derived shader wrapper files. That mixture is useful as a reference, but it is not an acceptable long-term compiler boundary.

Without a frozen interface split, the project risks hardening the wrong artifacts into the public compiler contract. In particular, the backend must not accidentally define itself around handwritten C files, handwritten mailbox logic, or benchmark-specific structs. The backend must instead consume a stable MLIR contract and emit a stable artifact contract rooted in C and vc4asm.

The backend is only successful if it ultimately emits a concrete artifact set based on generated C and generated vc4asm that can be built and executed on the Raspberry Pi VC4 environment.

## 4. Design goals

The backend must consume a canonical whole-program MLIR contract, not handwritten C source. That contract must make both host-side and device-side structure visible before backend-specific lowering.

The backend must preserve a clean distinction between logical GPU semantics and physical VC4 execution strategy. Source-level and canonical-IR kernels must remain logical. VC4-specific builtins, uniform ABI details, and persistent-worker execution structure must be introduced by lowering, not written by users.

The backend must emit concrete buildable outputs rooted in generated C and generated vc4asm. The emitted artifacts must be sufficient, together with a reusable handwritten runtime library, to build and run on the Pi.

The compiler must own kernel-specific launcher generation and device code generation. The runtime must own generic VC4 services such as mailbox interaction, QPU enablement, memory management, upload, and launch plumbing.

The design must support multi-level inspection. The project must be able to examine kernels at higher-level frontend IRs, at the MLIR `gpu` boundary, at backend-specific VC4 lowering levels, and at final emitted C and vc4asm.

The design must remain extensible. Narrow support for the current SAXPY slice is acceptable, but the interface split must not bake in artifacts that will need to be undone later.

## 5. Non-goals

This document does not define a single mandatory source language. Users may arrive through multiple frontends and programming models, as long as those frontends lower into the canonical MLIR contract defined here.

This document does not make handwritten C source the backend input. Handwritten C may be an upstream frontend source, a reference artifact, or a sample application, but it is not the formal VC4 backend boundary.

This document does not make the compiler responsible for the entire runtime stack or the entire application. The compiler must not absorb mailbox logic, generic GPU memory management, or benchmark harness code into its stable generated interface.

This document does not treat the current `saxpy.h` shape, with embedded arrays, code storage, and uniform blocks, as a long-term public boundary. That form is acceptable only as a reference artifact.

This document does not declare the backend complete when it can emit only internal backend IR or schematic vc4asm-like text. The required endpoint is buildable and runnable C plus vc4asm artifacts.

## 6. Architectural decision summary

The canonical backend input interface is a whole-program MLIR module centered on the `gpu` dialect. Host code remains visible as host IR. Device kernels remain visible as `gpu.module` and `gpu.func`. Host launch sites must reference kernel symbols and express launch semantics in MLIR before backend-specific lowering.

The generated artifact interface is a per-kernel artifact set rooted in `kernel.qasm`, `kernel_launch.c`, and `kernel_launch.h`. These files are the normative generated source-level outputs. Optional derived artifacts, such as assembled blobs or shader wrapper files, may exist for build convenience but are not the primary interface.

The compiler owns kernel-specific lowering, launcher generation, uniform packing logic, builtin introduction, and vc4asm emission. The runtime owns generic VC4 services and reusable launch support.

The VC4 backend lowers logical GPU kernels into a backend-specific representation that makes staged memory, physical execution builtins, and persistent-worker execution explicit. The current execution model uses `qpu_id`, `num_qpus`, a 16-wide SIMD chunk width, and a grid-stride loop.

## 7. The two frozen interfaces

### 7.1 Canonical backend input interface

The VC4 backend consumes a whole-program MLIR module.

That module must contain both host-side and device-side structure. Device kernels must appear in `gpu.module` as `gpu.func` operations. Host code must remain in host IR. Host launch sites must reference device kernel symbols and carry launch semantics in MLIR before backend-specific lowering.

The canonical backend input interface is therefore not a handwritten source file, not an ad hoc kernel launcher C API, and not handwritten vc4asm. It is a normalized MLIR contract in which the backend can see all of the following:

- the device kernel symbol
- the kernel argument list and types
- the host launch site
- the logical problem shape and launch semantics
- the host-device relationship between the launch site and the kernel body

For the current project direction, the canonical boundary is specifically the MLIR contract around the `gpu` dialect. Upstream frontends may lower into that contract in different ways, but the VC4 backend consumes the MLIR contract, not the original frontend language.

### 7.2 Generated artifact interface

The backend emits, per kernel, a concrete artifact set rooted in:

- `kernel.qasm`
- `kernel_launch.c`
- `kernel_launch.h`

These are the normative generated source-level outputs.

`kernel.qasm` is the device artifact. It must represent the lowered VC4 kernel semantics and must be suitable for assembly in the Pi VC4 environment.

`kernel_launch.c` and `kernel_launch.h` are the kernel-specific generated host launcher stub. They must expose only the semantic user/kernel arguments. They must not expose `qpu_id`, `num_qpus`, uniform packing, mailbox operations, or other device ABI details.

Optional build-derived artifacts may be produced from these files, such as assembled shader blobs or C wrappers around assembled output. Those derived artifacts are secondary. They must not become the primary compiler boundary.

## 8. Upstream programming models

Upstream programming models may vary. The project is expected to accept kernels that conceptually come from C-like host code with logical GPU kernels, CUDA-like kernel and launch models, higher-level compiler frontends such as Torch or XLA-derived stacks, or other source systems that eventually lower into the canonical MLIR `gpu` representation.

At the source level, users should conceptually write ordinary host-side logic, logical GPU kernels, and logical kernel launches by kernel symbol and problem size. Users should not write `qpu_id`, `num_qpus`, uniform packing code, mailbox calls, vc4asm, or hardware ABI details.

A representative upstream programming model looks like this:

```c
gpu_kernel
void saxpy(float *x, float *y, float a, size_t n) {
    size_t gid = gpu_global_id_x();
    if (gid < n)
        y[gid] = a * x[gid] + y[gid];
}

int main(void) {
    gpu_launch_1d(saxpy, n, x, y, a, n);
}
```

This is an upstream programming model example, not the formal backend input contract. The formal backend input contract is the MLIR form that this source model lowers into.

## 9. Whole-program IR model

The compiler must see host and device structure together before backend-specific emission. The whole-program MLIR model is the mechanism that provides that visibility.

The host side remains host IR. The device side remains `gpu.module` and `gpu.func`. The host launch site references the device kernel symbol and carries the launch semantics. That allows the backend to lower device code and host launch mechanics in a coordinated way.

A representative canonical whole-program MLIR shape is:

```mlir
module {
  func.func @main(%x: memref<?xf32>, %y: memref<?xf32>, %a: f32, %n: index) {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@saxpy
      blocks(%n, %c1, %c1) threads(%c1, %c1, %c1)
      args(%x, %y, %a, %n)
      : memref<?xf32>, memref<?xf32>, f32, index
    return
  }

  gpu.module @kernels {
    gpu.func @saxpy(%x: memref<?xf32>, %y: memref<?xf32>, %a: f32, %n: index) kernel {
      %gid = gpu.global_id x
      %inBounds = arith.cmpi ult, %gid, %n : index
      scf.if %inBounds {
        %xval = memref.load %x[%gid] : memref<?xf32>
        %yval = memref.load %y[%gid] : memref<?xf32>
        %mul = arith.mulf %a, %xval : f32
        %sum = arith.addf %mul, %yval : f32
        memref.store %sum, %y[%gid] : memref<?xf32>
      }
      gpu.return
    }
  }
}
```

The current backend slice may support only a narrow normalized form of this whole-program model, but this is the correct architectural shape. The backend must be able to see the host launch site, the kernel symbol, and the kernel body in one IR world before it emits C and vc4asm.

## 10. MLIR representation and lowering story

This backend exists to preserve an inspectable lowering path, not merely to perform final code generation. The project must support inspection across multiple levels, including high-level compiler IRs, MLIR `gpu`, backend-specific VC4 lowering, and final artifacts.

The intended lowering story is:

high-level source or frontend IR  
→ whole-program MLIR with host and `gpu` device structure  
→ backend-specific lowering into `vc4` plus selected upstream dialects  
→ generated C launcher artifacts and generated vc4asm  
→ build-derived artifacts and Pi executable

The `gpu` dialect is the canonical backend entry point because it preserves logical GPU semantics in an inspectable form. The VC4 backend then introduces the hardware-specific execution model during lowering rather than forcing users to write VC4-shaped kernels upstream.

The `vc4` dialect is not the canonical input boundary. It is a backend-specific lowered representation. Its purpose is to model staged memory movement, physical execution builtins, and VC4-oriented compute structure in a way that is still inspectable and not yet collapsed into raw assembler syntax.

For the current SAXPY slice, backend-specific lowering introduces the physical worker model explicitly:

```mlir
%qpu = vc4.get_builtin qpu_id : index
%num_qpus = vc4.get_builtin num_qpus : index
%c16 = arith.constant 16 : index
%base = arith.muli %qpu, %c16 : index
%stride = arith.muli %num_qpus, %c16 : index

scf.for %i = %base to %n step %stride {
  // staged VC4 memory movement and compute
}
```

This is the correct direction. Logical GPU indexing remains in the canonical input. Physical worker identity and the persistent-worker loop are introduced by the backend.

## 11. VC4-specific backend contract

The current backend-specific contract is defined by the lowered device-side representation and the device/runtime ABI that connects it to final code generation.

### 11.1 Logical versus physical execution

At the canonical MLIR input level, kernels remain logical and element-parallel. They use logical GPU constructs such as `gpu.global_id`.

At the VC4-lowered level, the backend introduces physical execution builtins and a persistent-worker execution strategy. The current builtins are:

- `qpu_id`
- `num_qpus`

These builtins are introduced internally by the backend. They are not source-level kernel arguments.

The current lowered execution model is:

- `base = qpu_id * 16`
- `stride = num_qpus * 16`
- explicit loop over 16-wide chunks

The current slice assumes `n % 16 == 0`. Tail handling is deferred.

### 11.2 Uniform ABI boundary

User kernel arguments and execution builtins remain distinct in IR.

User kernel arguments are represented as logical uniform accesses such as `vc4.get_uniform[i]`. Execution builtins are represented as explicit builtin accesses such as `vc4.get_builtin qpu_id`.

At runtime, both travel through the physical uniform stream. The general runtime uniform order is:

`arg[0], arg[1], ..., arg[K-1], builtin[0], builtin[1], ..., builtin[M-1]`

For the current prototype, the order is:

1. `x`
2. `y`
3. `a`
4. `n`
5. `qpu_id`
6. `num_qpus`

This ordering is normative for the current slice.

### 11.3 Sequential consumption

The physical uniform mechanism is a forward-only stream. Final code generation must consume uniforms sequentially with repeated `mov ..., unif` operations in exactly the documented runtime order.

`vc4.get_uniform[i]` is therefore a logical ABI position in IR, not a promise of physical random access storage. Likewise, `vc4.get_builtin` remains logically distinct in IR even if its physical transport is via reserved uniform suffix words.

### 11.4 Alpha handling

In the source and canonical MLIR input, alpha is scalar: `f32`.

In the current lowered VC4 IR, alpha may appear as a vector-typed uniform value because VC4 uniform reads are lane-broadcast and the target-side compute is vector-typed. This is a lowering detail. It does not change the runtime ABI. The runtime must pack one scalar alpha value, not a literal 16-lane vector payload.

### 11.5 Current lowered device operations

For the current slice, the backend-specific lowered device body uses the `vc4` dialect to represent:

- uniform reads
- builtin reads
- DMA load and store
- VPM read and write
- vector floating-point multiply and add

Loop and index arithmetic may remain in `arith`, `cf`, and `scf` while control flow remains simple.

## 12. Generated outputs

The backend emits concrete source-level outputs per kernel.

### 12.1 `kernel.qasm`

`kernel.qasm` is the normative generated device artifact. It must represent the lowered VC4 kernel semantics, consume the uniform stream in the documented order, and be suitable for assembly in the target Pi VC4 environment.

For the current SAXPY slice, the kernel must consume uniforms sequentially in this order:

```asm
mov rx, unif      ; x
mov ry, unif      ; y
mov ra, unif      ; a
mov rn, unif      ; n
mov rqpu, unif    ; qpu_id
mov rnum, unif    ; num_qpus
```

It must then derive `base` and `stride`, loop over 16-wide chunks, perform staged loads, compute SAXPY, and stage the result back to memory.

### 12.2 `kernel_launch.h`

`kernel_launch.h` declares the generated launcher interface for one kernel. That public generated interface must expose only semantic kernel arguments.

For the current SAXPY slice, the launcher shape is:

```c
int saxpy_launch(struct vc4_runtime *rt,
                 float *x,
                 float *y,
                 float a,
                 uint32_t n);
```

This interface is normative in spirit even if exact naming evolves. The key invariants are that it exposes only semantic kernel arguments and a runtime handle, and that it does not expose builtins or ABI internals.

### 12.3 `kernel_launch.c`

`kernel_launch.c` is the generated host-side kernel launcher. The compiler owns this file.

This file must:

- implement the kernel-specific launcher interface
- choose the active QPU count for the launch
- construct one per-QPU uniform payload
- pack that payload in the runtime order
- append `qpu_id` and `num_qpus` internally
- invoke the reusable handwritten runtime

This file must not inline generic mailbox logic or become a kernel-specific runtime implementation.

### 12.4 Derived artifacts

The build may derive additional artifacts such as assembled blobs or wrapper files. Files such as `saxpyshader.c` and `saxpyshader.h` belong to this category. They are build artifacts, not the primary source-level compiler output.

### 12.5 Buildability requirement

The backend is only successful if the generated `kernel.qasm`, `kernel_launch.c`, and any necessary build-derived artifacts can be combined with the handwritten runtime and build system to produce a runnable Pi artifact. Emitting only backend IR or only schematic qasm-like text is not sufficient.

## 13. Handwritten runtime contract

The handwritten runtime is separate from the compiler and is reusable across kernels.

The runtime owns generic VC4 services:

- mailbox interaction
- QPU enable and disable
- GPU memory allocation, locking, and addressing
- code upload
- generic launch plumbing
- generic support for launching a kernel with per-QPU uniform streams

The compiler must not own these services.

The runtime should know nothing about SAXPY specifically. It should accept generic code payloads, generic uniform payloads, and generic launch parameters. The generated launcher should adapt semantic kernel arguments into the form required by the runtime.

Runtime API names are not frozen by this document. Runtime responsibilities are frozen by this document.

## 14. End-to-end workflow

The end-to-end workflow is as follows.

A user writes a source program using an upstream programming model with ordinary host logic, logical GPU kernels, and logical kernel launches.

A frontend lowers that source program into the canonical whole-program MLIR contract. The host side remains visible as host IR. The device kernels remain visible in `gpu.module` and `gpu.func`. The host launch sites remain visible and reference kernel symbols.

The VC4 backend consumes that MLIR. It lowers device kernels from logical GPU semantics into a backend-specific VC4 representation that introduces physical execution builtins, staged-memory operations, and the persistent-worker loop. It also lowers host launch semantics to the generated launcher boundary.

The backend then emits, per kernel, `kernel.qasm`, `kernel_launch.c`, and `kernel_launch.h`.

The build system assembles `kernel.qasm` as needed, compiles the generated C launcher and the user host code, links them against the reusable runtime, and produces the Pi executable.

At execution time, the generated launcher packs the per-QPU uniform payloads, calls the handwritten runtime, and the runtime uploads and dispatches the kernel on the VC4 QPUs.

This workflow must end in a concrete buildable and runnable C-plus-vc4asm artifact set. That requirement is part of the architecture, not an optional finishing step.

## 15. SAXPY reference example, rewritten under the new split

Under the frozen architecture, the SAXPY example is split into three distinct layers.

### 15.1 User or sample application

The sample application remains ordinary host code. It does not know about `qpu_id`, `num_qpus`, uniform packing, or vc4asm. Its job is to prepare data and invoke the generated launcher.

```c
#include "saxpy_launch.h"

int main(void) {
    struct vc4_runtime rt;
    float *x = /* host-visible data */;
    float *y = /* host-visible data */;
    float a = 2.0f;
    uint32_t n = 1 << 20;

    vc4_runtime_init(&rt);
    saxpy_launch(&rt, x, y, a, n);
    vc4_runtime_shutdown(&rt);
    return 0;
}
```

### 15.2 Generated launcher

The generated launcher is kernel-specific compiler output.

```c
int saxpy_launch(struct vc4_runtime *rt,
                 float *x,
                 float *y,
                 float a,
                 uint32_t n);
```

Its implementation packs one uniform stream per QPU in the exact order:

`x, y, a, n, qpu_id, num_qpus`

It hides all builtin handling and physical launch details from the caller.

### 15.3 Generated device artifact

The generated device artifact is the qasm file. Conceptually, it begins by consuming uniforms sequentially:

```asm
mov rx, unif      ; x
mov ry, unif      ; y
mov ra, unif      ; a
mov rn, unif      ; n
mov rqpu, unif    ; qpu_id
mov rnum, unif    ; num_qpus
```

It then computes:

- `base = qpu_id * 16`
- `stride = num_qpus * 16`

and executes the loop:

- load a 16-wide chunk from `x`
- load a 16-wide chunk from `y`
- compute `a * x + y`
- store the 16-wide chunk back to `y`

This is the concrete target that the backend must ultimately emit for the current slice.

## 16. Mapping of current reference files into the new architecture

`3-test-saxpy.c` is classified as a user or sample application. It is not compiler output.

`saxpy.qasm` is classified as the device artifact and is the correct kind of file for the backend to generate.

`saxpy.c` is currently mixed. Under the frozen architecture it must be interpreted as containing two conceptual pieces: generated kernel-specific launcher logic and generic runtime logic that should move into the reusable handwritten runtime.

`saxpy.h` is not the long-term public compiler interface in its current form. Its large `struct saxpy_gpu`, embedded arrays, code storage, and uniform blocks are acceptable in a reference artifact but must not become the stable public boundary.

`saxpyshader.c` and `saxpyshader.h` are build or assembly artifacts. They are not the primary source-level compiler outputs.

## 17. Hard invariants

The formal VC4 backend input must be whole-program MLIR. It must not be handwritten C, handwritten vc4asm, or a benchmark-specific C launcher API.

Users must not write `qpu_id`, `num_qpus`, uniform packing logic, mailbox calls, or vc4asm in the source programming model that feeds the backend.

The canonical MLIR input must preserve logical GPU semantics. Physical worker builtins and persistent-worker loop structure must be introduced by backend lowering.

Kernel arguments must occupy the leading logical uniform positions in source signature order. Builtins must occupy a fixed reserved suffix in the runtime uniform stream. For the current slice, the order must be `x, y, a, n, qpu_id, num_qpus`.

Builtins must remain explicit and conceptually distinct in IR even if they are physically carried through the same uniform stream as kernel arguments.

Final code generation must treat the uniform mechanism as a sequential stream. It must not model physical uniforms as random-access storage.

The public generated launcher interface must expose only semantic kernel arguments plus a runtime handle. Builtins and ABI internals must remain internal.

The compiler must own generated launchers and vc4asm. The handwritten runtime must own generic VC4 services. The compiler must not absorb the entire runtime or the entire application.

The backend is only successful when it emits a C-plus-vc4asm artifact set that can build and execute on the Raspberry Pi VC4 environment.

## 18. Deferred questions / future extensions

Tail handling for non-multiples of 16 is deferred. The current slice may assume `n % 16 == 0`, but future work must extend the lowering and final artifacts without breaking the frozen interface split.

Broader launch models are deferred. The current slice uses a simple one-dimensional logical kernel and a persistent-worker lowering. More general launch configurations may be added later, but they must still lower through the canonical MLIR contract and into the same generated artifact split.

The exact shape of optional derived build artifacts is deferred. Wrapper files around assembled shaders may change, but they must remain secondary to `kernel.qasm` and `kernel_launch.c/h`.

Alternative generated launcher boundaries based on explicit runtime buffer handles are deferred. The current frozen generated interface is semantic-argument based. Any future buffer-handle specialization must not reintroduce builtins or hardware ABI details into the public generated interface.

Broader kernel coverage, richer control flow, masking, predication, and additional VC4 hardware features are deferred. They must extend the architecture, not replace it.

## 19. Conclusion

The VC4 backend must be defined around two frozen interfaces.

The input side is a whole-program MLIR contract centered on the `gpu` dialect, where host and device structure remain visible together and logical GPU semantics remain intact.

The output side is a concrete per-kernel artifact set rooted in generated C and generated vc4asm, backed by a reusable handwritten runtime.

This split is the durable architecture for the project. It preserves inspectability from high-level compiler forms down to Pi-executable artifacts, keeps the backend boundary clean, prevents handwritten runtime details from leaking into the compiler interface, and makes the success criterion explicit: the backend must ultimately emit buildable and runnable C-plus-vc4asm artifacts for the Raspberry Pi VC4 environment.

### Immediate implementation guidance

Preserve the current handwritten SAXPY under `code/3-saxpy` as the working baseline and reference of last resort.

Create a compiler-owned reference target under `compiler/reference/saxpy/` that reflects the frozen split in this document: a `saxpy.qasm` device artifact, a `saxpy_launch.c`, a `saxpy_launch.h`, and a short note describing their ABI relationship.

Extract generic mailbox, memory, upload, and launch plumbing out of the kernel-specific handwritten reference and into a reusable runtime library boundary.

Treat the generated launcher boundary, not the current handwritten `saxpy.h`, as the public kernel-specific interface.

Then drive the compiler work toward that reference target: the MLIR `gpu` input should lower to backend-specific VC4 IR, and the backend should emit `saxpy.qasm` plus `saxpy_launch.c/h` that match the reference ABI and execution model exactly.
