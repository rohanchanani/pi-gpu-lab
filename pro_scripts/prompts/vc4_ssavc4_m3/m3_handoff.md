# M3 Handoff: SSAVC4 Dialect and Lowering to Scheduled VC4

M3 starts from the completed M2 backend. M2 accepts already-scheduled `vc4` QPU programs and emits QASM, manifest v2 metadata, `layout.json`, generated `kernel_launch.c/h`, shader arrays, candidate workdirs, and Pi hardware runs through the libpi-backed runtime. M3 must preserve that path.

The architecture is:

```text
future M4: MLIR gpu dialect
    ↓
M3: ssavc4 dialect
    ↓
completed M2: scheduled vc4 dialect
    ↓
VC4ArtifactEmitter.cpp
    ↓
QASM / manifest.json / layout.json / kernel_launch.c/h / shader arrays
    ↓
libpi VC4 runtime
    ↓
VideoCore IV hardware
```

The post-cleanup active `vc4` dialect is intentionally narrow: `vc4.module`, `vc4.func`, `vc4.qpu.ldi`, `vc4.qpu.sema`, `vc4.qpu.bundle`, `vc4.qpu.branch`, and metadata/attrs/enums required by those scheduled sink ops. The old structured `vc4` surface is removed and must stay removed.

SSAVC4 is a new target-specific SSA machine IR. It uses MLIR builtin scalar/vector value types for ordinary data (`i32`, `f32`, `vector<16xi32>`, `vector<16xf32>`), minimal custom SSAVC4 types for tokens/descriptors/flags, live scheduled-VC4 attrs/enums where applicable, and SSAVC4-owned attrs for descriptor concepts removed from `vc4`.

Launch/resource metadata is copied through unchanged as `vc4.launch_abi` and `vc4.resource` dictionaries. The scheduled output must be `domain = #vc4.execution_domain<qpu>` and `form = #vc4.function_form<scheduled>`.

After the early scaffold slices, feature bring-up is vertical:

1. Define the required op/type/attr shape.
2. Add parser/printer/verifier/effects tests.
3. Lower to scheduled `vc4.qpu.*` templates.
4. Run scheduled-output checks.
5. Generate M2-compatible artifacts.
6. Verify fixture semantics.

Important hazards from M2:

- Branch immediates and delay slots must be computed after final scheduled layout, not guessed during local lowering.
- Runtime logical request/block/warp identity is not physical QPU identity.
- Independent-vector and cooperative-block fixtures have different resource requirements.
- Reference bundles may contain old raw mailbox/runtime code and are not the current generated candidate architecture.
- M2 and later milestones are cumulative; final M3 acceptance must run the generic M2 verifier.

Detailed design source: `compiler/docs/codegen/ssavc4-ir-design-m3-post-cleanup.md`. Treat it as the ground-truth implementation design for M3.
