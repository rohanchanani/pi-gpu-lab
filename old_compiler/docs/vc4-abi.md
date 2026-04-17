# VC4 Prototype Kernel/Runtime ABI

This note captures the current prototype ABI witness for the first feature-bounded VC4 backend slice. It is intentionally narrow and describes the code as it exists today, not the final long-term public compiler boundary.

## Source-Side Kernel ABI

The representative source kernel shape for `--lower-gpu-to-vc4` is:

```mlir
gpu.func @saxpy(%x: memref<?xf32>, %y: memref<?xf32>, %a: f32, %n: index) kernel
```

The source ABI remains:

1. `%x: memref<?xf32>`
2. `%y: memref<?xf32>`
3. `%a: f32`
4. `%n: index`

This is the user-visible kernel ABI. It is not target-shaped, and it does not encode physical VC4 execution builtins as fake extra kernel arguments.

## Lowered IR ABI Distinction

After lowering, the IR keeps two concepts distinct:

- User kernel arguments become user uniforms, read in source argument order with `vc4.get_uniform`.
- Execution builtins remain explicit target-side builtins, read with `vc4.get_builtin`.

For the current prototype slice, the lowered IR uses:

- `vc4.get_uniform[0]` for `x`
- `vc4.get_uniform[1]` for `y`
- `vc4.get_uniform[2]` for `a`
- `vc4.get_uniform[3]` for `n`
- `vc4.get_builtin qpu_id`
- `vc4.get_builtin num_qpus`

Even if a later runtime physically supplies builtin values through reserved uniform slots, builtins must remain distinct from ordinary user uniforms in IR.

## Prototype Runtime Uniform Layout

The current prototype runtime-side uniform payload for the current slice is:

1. slot `0`: `x`
2. slot `1`: `y`
3. slot `2`: `a`
4. slot `3`: `n`
5. slot `4`: `qpu_id`
6. slot `5`: `num_qpus`

The concrete host/runtime-facing artifact for this layout lives at
`compiler/examples/saxpy_runtime_abi.h`.

Important distinction:

- Slots `0` through `3` correspond to user kernel arguments and are modeled in IR with `vc4.get_uniform`.
- Slots `4` and `5` are reserved runtime execution inputs. In IR they are modeled with `vc4.get_builtin`, not with `vc4.get_uniform`.

The physical uniform interface is sequential. A final emitted kernel-side
realization must consume this payload with repeated `mov ..., unif` in exactly
this order. The runtime payload should not be treated as random-access
physical storage.

This note defines the current prototype runtime layout for the current slice. It does not imply that the final runtime ABI is frozen. It should be treated as a temporary ABI witness while the backend is being stabilized around the normative artifact boundary in `backend_spec.md`.

The current prototype packs these values as 32-bit uniform words. In
particular, `n`, `qpu_id`, and `num_qpus` are physically carried as single
32-bit runtime words even though the source/lowered IR models them as `index`.
The current prototype slice therefore assumes these values fit in 32 bits.

## Alpha Handling

Source alpha remains a scalar `f32` argument in the source kernel ABI.

The current lowering materializes alpha in the target-side IR as:

```mlir
%a = vc4.get_uniform[2] : vector<16xf32>
```

This is a lowering detail for the current VC4 backend slice. It reflects the current assumption that uniform reads are lane-broadcast across the width-16 execution group, so the lowered VC4 compute path consumes alpha as a vector value without an explicit `vector.broadcast` op.

This does not change the source ABI: source alpha is still scalar.

On the host/runtime side, alpha is packed once as one scalar `f32` word. The
runtime does not materialize a literal 16-lane vector payload for alpha.

## Persistent-Worker Execution Model

The current lowered execution strategy is a persistent-worker / grid-stride form over 16-wide chunks:

```mlir
%qpu = vc4.get_builtin qpu_id : index
%numQpus = vc4.get_builtin num_qpus : index
%base = arith.muli %qpu, %c16 : index
%stride = arith.muli %numQpus, %c16 : index
scf.for %iv = %base to %n step %stride
```

For the current prototype slice:

- `base = qpu_id * 16`
- `stride = num_qpus * 16`
- each loop iteration handles one 16-element chunk
- `n` must be a multiple of 16
- tail handling is not implemented yet

This is where logical GPU indexing is mapped onto physical VC4 workers in the current prototype.

## Current Integration Boundary

The repo already contains a handwritten SAXPY runtime path under
`code/3-saxpy/`, but it is not ABI-compatible with this compiler prototype yet.
That handwritten path is still useful as a local assembler/runtime reference.

For the compiler slice today:

- treat `compiler/examples/saxpy_runtime_abi.h` as a concrete prototype ABI
  witness for this slice, not as the long-term public compiler boundary
- treat `code/3-saxpy/` as a reference for machine structure and local VC4
  bring-up, not as a claim of binary compatibility with compiler output
