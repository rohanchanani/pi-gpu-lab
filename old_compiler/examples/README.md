# Compiler Examples

## `saxpy_runtime_abi.h`

`saxpy_runtime_abi.h` is a concrete ABI witness for the first feature-bounded
VC4 backend slice. It does not add a runtime, and it is not the long-term
public compiler boundary. It only freezes the prototype per-QPU uniform payload
that the current lowered IR expects.

The current physical runtime payload order for one launched QPU is:

1. `x`
2. `y`
3. `a`
4. `n`
5. `qpu_id`
6. `num_qpus`

The order matters because the physical VC4 uniform interface is a forward-only
stream. A final emitted kernel must realize this payload with repeated
`mov ..., unif` in exactly this order. This is not random-access physical
storage.

Important distinction:

- `x`, `y`, `a`, and `n` are source kernel arguments and lower to
  `vc4.get_uniform[0..3]`.
- `qpu_id` and `num_qpus` are execution builtins in IR, even if a runtime
  later delivers them through reserved uniform slots.
- `a` is packed as one scalar `f32` uniform word on the host/runtime side.
  The lowered VC4 IR reads it as `vector<16xf32>` as a target-side lowering
  detail because the current prototype assumes lane-broadcast uniform reads.

## Prototype Integration Contract

The current compiler can already produce intermediate inspection-oriented VC4
assembly text:

```bash
compiler/build/bin/vc4-opt --lower-gpu-to-vc4 input.mlir \
  | compiler/build/bin/vc4-translate --mlir-to-vc4asm
```

This inspection output is an intermediate checkpoint only. The normative target
boundary remains generated `kernel.qasm`, `kernel_launch.c`, and
`kernel_launch.h` as described in `compiler/docs/backend_spec.md`.

When a future emitted VC4 program adopts this prototype ABI, it should be
paired with one `vc4::examples::SaxpyUniformBlock` per launched QPU, with:

- shared `x`, `y`, `a`, and `n`
- per-QPU `qpu_id`
- shared `num_qpus`

The expected physical consumption shape is:

```text
mov ..., unif    ; x
mov ..., unif    ; y
mov ..., unif    ; a
mov ..., unif    ; n
mov ..., unif    ; qpu_id
mov ..., unif    ; num_qpus
```

## Relation To The Existing Handwritten Path

The repo already has a handwritten VC4 SAXPY path at:

```bash
cd code/3-saxpy
bash run.sh
```

That path is useful as a machine-structure reference and as the obvious local
assembler/runtime path in this repo.

It is not a drop-in runtime for the compiler prototype ABI yet:

- the handwritten `code/3-saxpy/saxpy.qasm` currently uses a different uniform
  order
- it also uses an iteration-count convention that is specific to that kernel

So the compiler-side ABI artifact should be treated as a prototype witness for
the current slice, not as a claim that the current handwritten runtime already
consumes compiler-emitted output unchanged and not as the final public compiler
interface.
