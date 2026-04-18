# VC4 MLIR Dialect Specification (compute/QPU scope)

## 1. Purpose and pipeline position

This document is the implementation specification for the `vc4` dialect milestone.

The intent is:

```text
upstream whole-program MLIR
  └─ gpu.module / gpu.func / gpu.launch_func / generic MLIR IR
      └─ vc4.module / vc4.func (structured form)
          └─ vc4.func (scheduled QPU form)
              └─ qasm + launcher generation (later, out of scope here)
```

`gpu.module` and `gpu.func` remain the upstream source representation. `vc4.module` and `vc4.func`
are *downstream target-specific IR* introduced by lowering from the upstream GPU dialect family.
They do **not** replace the upstream GPU dialect.

This dialect is intentionally **hardware-facing** and **compute/QPU focused**.

## 2. Scope adjustments for this milestone

The original design included the full graphics-side hardware surface. That is now intentionally pruned.

### Included in this milestone

- QPU structured compute IR
- QPU scheduled / emission-adjacent IR
- uniforms
- TMU (including async/direct-address usage)
- SFU
- VPM
- VDR / VDW DMA
- semaphores
- inter-QPU mutex
- host interrupt
- thread switching / program end
- user-program queue submission and relevant V3D scheduler / system configuration hooks

### Explicitly out of scope for this milestone

Do **not** implement any of the following in the first `vc4` dialect milestone:

- TLB / tile buffer ops
- scoreboard ops
- stencil ops
- fragment shader startup registers
- fragment/pixel builtins
- varyings interpolation / varying FIFO ops
- GL / NV / VG shader-state records
- control-list records
- binning / rendering list IR

If these features are ever added later, they should be added consciously as a new extension of the dialect,
not reintroduced accidentally during this milestone.

## 3. Design principles

1. **Two canonical strata inside one dialect**
   - A **structured** SSA form for target-specific semantics.
   - A **scheduled QPU form** for actual instruction encoding and final emission adjacency.

2. **Hardware truth wins**
   - QPU instruction bundling, branch delay slots, load-immediate, and semaphore instructions are modeled explicitly.
   - TMU/VPM/DMA/SFU are represented as semantic ops in structured form and as encoding-level ops in scheduled form.

3. **Do not model source metaprogramming as runtime IR**
   - vc4asm directives such as `.macro`, `.rep`, `.include`, `.func`, and source helper macros are not dialect ops.

4. **Stay close to MLIR conventions**
   - Use symbols for target containers and device functions.
   - Use builtin vector types for SIMD values.
   - Use custom resource types only where the value is not an ordinary scalar/vector.

5. **Prepare for later lowering and codegen without implementing them yet**
   - No lowering from `gpu` in this milestone.
   - No qasm emission in this milestone.
   - No launcher C generation in this milestone.

## 4. Canonical forms

## 4.1 Structured form

Structured form is the normal target IR used after lowering from upstream GPU-ish IR and before scheduling.

Characteristics:

- SSA values
- analyzable
- scheduler-agnostic
- value-centric TMU/VPM/DMA/SFU/uniform operations
- may contain CFG with `vc4.cf.branch`
- may contain `vc4.return`

Allowed ops in structured form are the ops listed in sections 10.1 through 10.8 below, excluding `vc4.qpu.*`.

## 4.2 Scheduled QPU form

Scheduled form is the final low-level target IR before qasm emission.

Characteristics:

- explicit instruction words
- explicit hardware fields
- direct modeling of QPU instruction forms
- no high-level SSA arithmetic ops
- branch delay slots are explicit
- register-file / mux / small-immediate / signal legality is verified here

Allowed ops in scheduled form are:

- `vc4.qpu.bundle`
- `vc4.qpu.branch`
- `vc4.qpu.ldi`
- `vc4.qpu.sema`

plus structural container ops (`vc4.module`, `vc4.func`), and optionally comments/locations as normal MLIR metadata.

Structured and scheduled forms must not be mixed in the same `vc4.func`.

## 5. Symbol structure

## 5.1 `vc4.module`

Top-level VC4 target container. Analogous in role to `gpu.module`, but target-specific.

Contents allowed:

- `vc4.func`
- later, if ever needed, additional VC4-specific symbols

## 5.2 `vc4.func`

Target-specific device function.

Recommended attributes:

- `kernel` unit attr: marks entry-point kernels produced from upstream `gpu.func kernel`
- `threading` enum attr: `single | threadable`
- `form` enum attr: `structured | scheduled`
- optional target config attrs or metadata as needed later

Use `FunctionOpInterface`.

## 6. Types

Use builtin MLIR types for ordinary data whenever possible:

- `i1`, `i8`, `i16`, `i32`, `f16`, `f32`
- `index`
- `vector<16xi32>`, `vector<16xf32>`, and other builtin 16-lane vector types

Add only the following custom types in the first milestone:

1. `!vc4.async.token`
2. `!vc4.tmu.desc`
3. `!vc4.vpm.desc`
4. `!vc4.dma.desc`

### 6.1 Type invariants

- All QPU value ops operate on scalars or 16-lane vectors.
- The dialect must reject unsupported lane counts for QPU arithmetic/value-shape ops.
- Descriptor types are opaque semantic objects. They are not pointers, not memories, and not raw integers.

## 7. Attributes and enums

The exact C++/ODS naming can vary, but the dialect must expose the following enum spaces.

## 7.1 Function/container enums

- `VC4ThreadingMode`: `single`, `threadable`
- `VC4FunctionForm`: `structured`, `scheduled`

## 7.2 Builtin enums

- `VC4BuiltinKind`
  - `elem_num`
  - `qpu_num`

## 7.3 ALU enums

- `VC4AddOpcode`
  - `nop`
  - `fadd`
  - `fsub`
  - `fmin`
  - `fmax`
  - `fminabs`
  - `fmaxabs`
  - `ftoi`
  - `itof`
  - `add`
  - `sub`
  - `shr`
  - `asr`
  - `ror`
  - `shl`
  - `min`
  - `max`
  - `and`
  - `or`
  - `xor`
  - `not`
  - `clz`
  - `v8adds`
  - `v8subs`

- `VC4MulOpcode`
  - `nop`
  - `fmul`
  - `mul24`
  - `v8muld`
  - `v8min`
  - `v8max`
  - `v8adds`
  - `v8subs`

- `VC4Cond`
  - `never`
  - `always`
  - `zs`
  - `zc`
  - `ns`
  - `nc`
  - `cs`
  - `cc`

## 7.4 Pack/unpack and immediate enums

- `VC4LoadImmMode`
  - `splat32`
  - `per_elem_i2`
  - `per_elem_u2`

- `VC4RegfileAUnpackMode`
  - `none`
  - `f16a_or_i16a`
  - `f16b_or_i16b`
  - `replicate_8d`
  - `color8a`
  - `color8b`
  - `color8c`
  - `color8d`

- `VC4RegfileAPackMode`
  - `none`
  - `to_16a`
  - `to_16b`
  - `to_8888`
  - `to_8a`
  - `to_8b`
  - `to_8c`
  - `to_8d`
  - `sat32`
  - `sat16a`
  - `sat16b`
  - `sat8888`
  - `sat8a`
  - `sat8b`
  - `sat8c`
  - `sat8d`

- `VC4R4UnpackMode`
  - `none`
  - `f16a`
  - `f16b`
  - `replicate_8d`
  - `color8a`
  - `color8b`
  - `color8c`
  - `color8d`

- `VC4MulPackMode`
  - `none`
  - `to_8888`
  - `to_8a`
  - `to_8b`
  - `to_8c`
  - `to_8d`

## 7.5 TMU enums

- `VC4TMUUnit`: `tmu0`, `tmu1`
- `VC4TMUMode`: `direct`, `texture2d`, `cubemap`
- `VC4TextureType`
  - `rgba8888`
  - `rgbx8888`
  - `rgba4444`
  - `rgba5551`
  - `rgb565`
  - `luminance`
  - `alpha`
  - `lumalpha`
  - `etc1`
  - `s16f`
  - `s8`
  - `s16`
  - `bw1`
  - `a4`
  - `a1`
  - `rgba64`
  - `rgba32r`
  - `yuyv422r`
- `VC4MagFilter`: `linear`, `nearest`
- `VC4MinFilter`
  - `linear`
  - `nearest`
  - `near_mip_near`
  - `near_mip_lin`
  - `lin_mip_near`
  - `lin_mip_lin`
- `VC4WrapMode`: `repeat`, `clamp`, `mirror`, `border`
- `VC4TMUReadPart`
  - `raw32`
  - `rgba8888`
  - `rg1616`
  - `ba1616`

## 7.6 SFU enums

- `VC4SFUKind`
  - `recip`
  - `recipsqrt`
  - `exp`
  - `log`

## 7.7 VPM/DMA enums

- `VC4VPMDescKind`: `read`, `write`
- `VC4VPMOrientation`: `horizontal`, `vertical`
- `VC4VPMLaneMode`: `packed`, `laned`
- `VC4VPMElemWidth`: `w8`, `w16`, `w32`

- `VC4DMADescKind`: `load`, `store`
- `VC4DMABlockMode`: `row_row`, `packed_rows`
- additional DMA setup enums may be introduced if needed, but the descriptor op must preserve all hardware fields necessary to model:
  - width/start-byte-halfword selection
  - mpitch/vpitch
  - nrows/rowlen
  - units/depth
  - orientation
  - VPM base
  - stride / extended stride

## 7.8 Sync/thread/system enums

- `VC4SemaphoreMode`: `acquire`, `release`
- `VC4MutexMode`: `acquire`, `release`
- `VC4ThreadSwitchMode`: `switch`, `last_switch`

## 7.9 Structured branch enums

- `VC4BranchCond`
  - `all_z_set`
  - `all_z_clear`
  - `any_z_set`
  - `any_z_clear`
  - `all_n_set`
  - `all_n_clear`
  - `any_n_set`
  - `any_n_clear`
  - `all_c_set`
  - `all_c_clear`
  - `any_c_set`
  - `any_c_clear`
  - `always`

## 7.10 Scheduled QPU enums

The sink ops must also expose enums for low-level instruction fields.

At minimum:

- `VC4QPUSignal`
  - `bkpt`
  - `none`
  - `thrsw`
  - `thrend`
  - `last_thread_switch`
  - `small_imm`
  - `load_imm`
  - `branch`
  - `ldtmu0`
  - `ldtmu1`

Plus any additional compute-relevant signals retained in the compute-focused scope.

Note: compute-focused scope intentionally omits tile-buffer / scoreboard signals from the first milestone, but omitted
members do not imply renumbering of the retained hardware signal encodings.

## 8. Traits, interfaces, and side-effect modeling

## 8.1 Required interfaces

Implement these in the first milestone:

- `FunctionOpInterface` for `vc4.func`
- `SymbolOpInterface` for `vc4.module` and `vc4.func` as appropriate
- `MemoryEffectOpInterface` for side-effecting ops

Custom VC4-specific interfaces are optional in this milestone. Do **not** block the milestone on them.

## 8.2 Custom effect resources

Use `MemoryEffectOpInterface` with custom resources instead of collapsing everything into `UnknownEffect`.

Recommended resources:

- `UniformStream`
- `MainMemory`
- `TMUReq0`
- `TMURcv0`
- `TMUReq1`
- `TMURcv1`
- `SFU`
- `VPMReadFIFO`
- `VPMWriteFIFO`
- `VDR`
- `VDW`
- `Mutex`
- `Semaphore`
- `HostIRQ`
- `QPUScheduler`
- `V3DSystem`

## 8.3 Verification layers

Use three verification layers:

1. **Local op verification**
   - type/arity/enum legality
   - operand/result consistency
   - impossible attribute combinations

2. **Function-form verification**
   - structured vs scheduled op segregation
   - `thread_switch` legality against `threading`
   - scheduled functions contain only scheduled ops

3. **Encoding / schedule-local verification**
   - branch delay slot count
   - per-instruction low-level legality for `vc4.qpu.*`
   - sequence-sensitive hazards may be deferred to later scheduling/legalization passes and do not need to be fully solved in the first milestone

## 9. General implementation conventions

## 9.1 TableGen split

Recommended file split:

- `VC4Dialect.td`
- `VC4Enums.td`
- `VC4Types.td`
- `VC4StructuredOps.td`
- `VC4MemoryOps.td`
- `VC4SystemOps.td`
- `VC4QPUOps.td`

## 9.2 C++ split

Recommended source split:

- `VC4Dialect.cpp`
- `VC4Types.cpp`
- `VC4Enums.cpp` if needed
- `VC4Ops.cpp` for shared logic
- `VC4StructuredOps.cpp`
- `VC4MemoryOps.cpp`
- `VC4SystemOps.cpp`
- `VC4QPUOps.cpp`
- `VC4SideEffects.cpp` if resources are separated

## 9.3 Assembly syntax goals

- Use declarative assembly format wherever possible.
- Use custom parsers/printers only for:
  - sink-level `vc4.qpu.*` ops
  - any op whose ergonomic syntax cannot be expressed reasonably with ODS

## 10. Operation set

The following op list is the implementation target for the milestone.

---

## 10.1 Program/container ops

### 1. `vc4.module`

**Kind**: symbol/container op

**Operands**: none  
**Results**: none  
**Regions**: one region containing symbols

**Attributes**
- optional target metadata attrs such as:
  - `num_slices`
  - `qpus_per_slice`
  - `tmus_per_slice`
  - `num_semaphores`
  - `vpm_size_kb`
  - `revision`
  - `hdr_support`
- all target metadata attrs are optional in the first milestone

**Semantics**
- container for VC4-specific device IR
- target-specific analogue to `gpu.module`

**Verifier**
- only legal nested ops appear inside
- symbol table validity

**Representation**
- structured target IR concept

**Expected lifetime**
- survives to later binary/launcher work

---

### 2. `vc4.func`

**Kind**: symbol/function op

**Operands**: function arguments  
**Results**: function results  
**Regions**: one body region

**Attributes**
- `kernel` unit attr (optional)
- `threading` = `single | threadable`
- `form` = `structured | scheduled`

**Semantics**
- target-specific device function
- `kernel` marks externally launched entry points
- `form` declares whether the function is in structured or scheduled QPU form

**Verifier**
- structured form must not contain `vc4.qpu.*`
- scheduled form must contain only `vc4.qpu.*` plus structural container ops/terminators if absolutely needed
- `thread_switch` is illegal unless `threading = threadable`

**Representation**
- structured target IR concept

**Expected lifetime**
- survives through the entire target pipeline

---

### 3. `vc4.return`

**Kind**: terminator

**Operands**: optional return values  
**Results**: none

**Attributes**
- none

**Semantics**
- structured-form return
- entry-point kernels will later normalize to `vc4.program_end` sequences

**Verifier**
- only legal in structured form
- operand/result types must match enclosing function signature

**Representation**
- structured target IR concept

**Expected lifetime**
- normalized before final emission

---

## 10.2 Builtin and uniform ops

### 4. `vc4.builtin`

**Operands**: none  
**Results**: one value

**Attributes**
- `kind` = `elem_num | qpu_num`

**Semantics**
- materializes QPU builtins available as register-mapped inputs

**Verifier**
- result type must be `i32` or `vector<16xi32>`
- if vector type is used, semantics are explicit lane replication by lowering convention

**Representation**
- direct hardware, structured form

**Expected lifetime**
- lowers to QPU register reads / source muxing

---

### 5. `vc4.uniform.read`

**Operands**: none  
**Results**: one value

**Attributes**
- optional `type_hint`

**Semantics**
- consumes one element from the uniform stream and returns it

**Verifier**
- result type must be scalar 32-bit or a legal VC4 vector view chosen by lowering convention

**Side effects**
- reads `UniformStream`

**Representation**
- direct hardware stream read

**Expected lifetime**
- survives until bundle lowering

---

### 6. `vc4.uniform.seek`

**Operands**: one scalar address  
**Results**: none

**Attributes**
- `relative` = `false | true`

**Semantics**
- resets or adjusts the uniform base pointer using SIMD element 0 semantics

**Verifier**
- operand must be scalar integer-like
- no result

**Side effects**
- writes `UniformStream`

**Representation**
- direct hardware pointer write

**Expected lifetime**
- survives until bundle lowering

---

## 10.3 Structured arithmetic and value-shape ops

### 7. `vc4.alu.add`

**Operands**
- one or two operands depending on opcode

**Results**
- one result

**Attributes**
- `op` : `VC4AddOpcode`
- `cond` : `VC4Cond`
- optional `set_flags` : unit/bool

**Semantics**
- structured ADD-pipe operation

**Verifier**
- opcode/arity/type pair must be legal
- unary ops (`not`, `clz`, conversion ops) have one operand
- binary ops have two operands
- types must be scalar/vector forms legal for the opcode

**Representation**
- direct hardware opcode family in SSA form

**Expected lifetime**
- bundles into `vc4.qpu.bundle`

---

### 8. `vc4.alu.mul`

**Operands**
- one or two operands depending on opcode

**Results**
- one result

**Attributes**
- `op` : `VC4MulOpcode`
- `cond` : `VC4Cond`
- optional `set_flags` : unit/bool

**Semantics**
- structured MUL-pipe operation

**Verifier**
- opcode/arity/type pair must be legal

**Representation**
- direct hardware opcode family in SSA form

**Expected lifetime**
- bundles into `vc4.qpu.bundle`

---

### 9. `vc4.mov`

**Operands**
- one source value

**Results**
- one result

**Attributes**
- optional `preferred_encoding` = `auto | small_imm | alu | ldi`

**Semantics**
- identity move in the vc4asm sense
- does not force one exact hardware encoding

**Verifier**
- source/result types must match or be bitcast-compatible under later lowering policy

**Representation**
- assembler-inspired structured convenience op

**Expected lifetime**
- normalized before final emission

---

### 10. `vc4.load_imm`

**Operands**: none  
**Results**: one result

**Attributes**
- `mode` : `VC4LoadImmMode`
- immediate payload

**Semantics**
- force a real QPU load-immediate style constant materialization

**Verifier**
- payload shape must match mode
- result type must be compatible with the mode

**Representation**
- direct hardware concept, structured form

**Expected lifetime**
- often lowers directly to `vc4.qpu.ldi`

---

### 11. `vc4.pack`

**Operands**
- one value

**Results**
- one value

**Attributes**
- one of the legal pack mode enums for the chosen path

**Semantics**
- explicit data packing / narrowing / saturation / color-pack operation

**Verifier**
- source/result type and mode must match one legal hardware pack interpretation

**Representation**
- structured value-shaping op backed by real pack bits

**Expected lifetime**
- usually fuses into low-level instruction fields

---

### 12. `vc4.unpack`

**Operands**
- one value

**Results**
- one value

**Attributes**
- one of the legal unpack mode enums

**Semantics**
- explicit unpack / extend / float16 / color conversion operation

**Verifier**
- mode/source/result combination must be legal

**Representation**
- structured value-shaping op backed by real unpack bits

**Expected lifetime**
- usually fuses into low-level instruction fields

---

### 13. `vc4.rotate`

**Operands**
- one vector input
- optional scalar amount operand

**Results**
- one vector result

**Attributes**
- optional immediate rotate amount

**Semantics**
- 16-lane horizontal rotate of a vector

**Verifier**
- input/result types must match
- vector length must be 16
- either an immediate amount or one scalar amount operand is present, not both
- amount must be representable as VC4 rotate immediate or later legalizable form

**Representation**
- partly direct hardware, partly structured convenience

**Expected lifetime**
- fuses or lowers into small-immediate rotate usage

---

### 14. `vc4.read`

**Operands**
- one source handle/value

**Results**
- optional value (implementation choice: either no result and pure side-effect, or one result if the design prefers explicit use)

**Attributes**
- none

**Semantics**
- vc4asm-style pseudo read that allocates a source read without consuming an ALU op

**Verifier**
- source must denote a legal readable source in the structured lowering model

**Representation**
- assembler-level pseudo concept

**Expected lifetime**
- normalized before final emission

**Implementation note**
- It is acceptable to model this as producing no SSA result in the first milestone, because its main purpose is scheduling/source allocation intent.

---

## 10.4 TMU and SFU ops

### 15. `vc4.tmu.descriptor`

**Operands**
- optional scalar config operands

**Results**
- one `!vc4.tmu.desc`

**Attributes**
- `mode` = `direct | texture2d | cubemap`
- constant texture setup fields when known:
  - base
  - type
  - mip levels
  - width / height
  - filters
  - wrap modes
  - flipY
  - cube-map stride
  - child-image fields
  - bias-mode flags

**Semantics**
- immutable semantic description of a TMU configuration

**Verifier**
- field combinations must match mode
- direct mode must reject texture-only fields

**Side effects**
- none (descriptor construction itself is pure)

**Representation**
- structured target IR concept backed by hardware config words

**Expected lifetime**
- survives until TMU request lowering

---

### 16. `vc4.tmu.request`

**Operands**
- direct mode: address operand, optional descriptor
- texture2d mode: `s`, optional `t`, optional `b`, descriptor
- cubemap mode: `s`, `t`, `r`, optional `b`, descriptor

**Results**
- optional `!vc4.async.token`

**Attributes**
- `unit` = `tmu0 | tmu1`

**Semantics**
- enqueue a TMU request into the TMU request FIFO

**Verifier**
- operand set must match descriptor mode
- direct mode requests must be 32-bit-address compatible
- unit must be valid

**Side effects**
- writes `TMUReq0` or `TMUReq1`
- may read `MainMemory` semantically via TMU

**Representation**
- structured op very close to hardware

**Expected lifetime**
- survives until bundle lowering

---

### 17. `vc4.tmu.read`

**Operands**
- optional `!vc4.async.token`

**Results**
- one value

**Attributes**
- `unit` = `tmu0 | tmu1`
- `part` = `raw32 | rgba8888 | rg1616 | ba1616`

**Semantics**
- read a completed TMU result from the receive FIFO

**Verifier**
- token, if present, must be compatible with TMU use
- result type must match `part`

**Side effects**
- reads `TMURcv0` or `TMURcv1`

**Representation**
- direct hardware receive/readback

**Expected lifetime**
- survives until bundle lowering

---

### 18. `vc4.tmu.noswap`

**Operands**
- optional boolean-like input

**Results**
- none

**Attributes**
- optional `disable` bool if no operand is used

**Semantics**
- control TMU automatic swapping for the current thread

**Verifier**
- exactly one of operand or boolean attr form may be chosen

**Side effects**
- writes `V3DSystem` or a dedicated TMU swap resource

**Representation**
- direct hardware control write

**Expected lifetime**
- survives until bundle lowering

---

### 19. `vc4.sfu.issue`

**Operands**
- one input value

**Results**
- none

**Attributes**
- `kind` = `recip | recipsqrt | exp | log`

**Semantics**
- issue an SFU operation

**Verifier**
- operand type must be legal scalar/vector 32-bit quantity

**Side effects**
- writes `SFU`

**Representation**
- direct hardware write

**Expected lifetime**
- survives until schedule/bundle lowering

---

### 20. `vc4.sfu.read`

**Operands**
- none

**Results**
- one value

**Attributes**
- none

**Semantics**
- read back an SFU result after the required latency

**Verifier**
- result type must be legal

**Side effects**
- reads `SFU`

**Representation**
- direct hardware readback

**Expected lifetime**
- survives until bundle lowering

---

## 10.5 VPM and DMA ops

### 21. `vc4.vpm.desc`

**Operands**
- optional scalar setup operands

**Results**
- one `!vc4.vpm.desc`

**Attributes**
- `kind` = `read | write`
- `orientation` = `horizontal | vertical`
- `lane_mode` = `packed | laned`
- `elem_width` = `w8 | w16 | w32`
- `addr`
- `stride`
- for reads: `num_vectors`

**Semantics**
- immutable descriptor for VPM block read/write setup

**Verifier**
- field ranges and combinations must match hardware formats

**Side effects**
- none

**Representation**
- structured descriptor corresponding to setup words

**Expected lifetime**
- survives until VPM read/write lowering

---

### 22. `vc4.vpm.read`

**Operands**
- one `!vc4.vpm.desc`

**Results**
- one vector/scalar value

**Attributes**
- none

**Semantics**
- consume one vector from a configured VPM read stream

**Verifier**
- descriptor kind must be `read`

**Side effects**
- reads `VPMReadFIFO`

**Representation**
- direct hardware effect, structured form

**Expected lifetime**
- survives until bundle lowering

---

### 23. `vc4.vpm.write`

**Operands**
- one `!vc4.vpm.desc`
- one value

**Results**
- none

**Attributes**
- none

**Semantics**
- write one vector to the configured VPM write stream

**Verifier**
- descriptor kind must be `write`

**Side effects**
- writes `VPMWriteFIFO`

**Representation**
- direct hardware effect, structured form

**Expected lifetime**
- survives until bundle lowering

---

### 24. `vc4.dma.desc`

**Operands**
- optional scalar setup operands

**Results**
- one `!vc4.dma.desc`

**Attributes**
- `kind` = `load | store`
- exact setup fields sufficient to encode VDR / VDW basic and extended formats

**Semantics**
- immutable DMA descriptor for VDR or VDW

**Verifier**
- field combinations must match chosen DMA kind

**Side effects**
- none

**Representation**
- structured descriptor backed by hardware setup words

**Expected lifetime**
- survives until DMA start lowering

---

### 25. `vc4.dma.start`

**Operands**
- one `!vc4.dma.desc`
- one base address

**Results**
- optional `!vc4.async.token`

**Attributes**
- none

**Semantics**
- start a DMA load/store

**Verifier**
- address must be scalar integer-like
- descriptor kind must be load or store

**Side effects**
- writes `VDR` or `VDW`
- accesses `MainMemory`

**Representation**
- direct hardware start action, structured form

**Expected lifetime**
- survives until bundle lowering or launcher lowering

---

### 26. `vc4.dma.status`

**Operands**
- none

**Results**
- one status value

**Attributes**
- `kind` = `load | store`

**Semantics**
- poll DMA busy / status

**Verifier**
- result type must be integer-like

**Side effects**
- reads `VDR` or `VDW`

**Representation**
- direct hardware poll

**Expected lifetime**
- may survive or fold

---

### 27. `vc4.dma.wait`

**Operands**
- optional `!vc4.async.token`

**Results**
- none

**Attributes**
- optional `kind` = `load | store` when no token is supplied

**Semantics**
- wait until DMA completes

**Verifier**
- exactly one of token or kind must be available in the operation syntax/design

**Side effects**
- reads `VDR` or `VDW`

**Representation**
- direct hardware wait or structured wait

**Expected lifetime**
- survives until low-level lowering

---

## 10.6 Synchronization and thread-control ops

### 28. `vc4.mutex`

**Operands**
- none

**Results**
- optional result on acquire (implementation may expose the read value or ignore it)

**Attributes**
- `mode` = `acquire | release`

**Semantics**
- inter-QPU hardware mutex

**Verifier**
- no extra invariants beyond attr legality

**Side effects**
- reads/writes `Mutex`

**Representation**
- direct hardware I/O mapped mutex

**Expected lifetime**
- survives until bundle lowering

---

### 29. `vc4.semaphore`

**Operands**
- none

**Results**
- optional result

**Attributes**
- `mode` = `acquire | release`
- `id` = semaphore number

**Semantics**
- acquire/decrement or release/increment one of the 16 hardware semaphores

**Verifier**
- `id` must be representable in 4 bits if constant

**Side effects**
- reads/writes `Semaphore`

**Representation**
- direct hardware semaphore action

**Expected lifetime**
- may lower directly to `vc4.qpu.sema`

---

### 30. `vc4.host_interrupt`

**Operands**: none  
**Results**: none

**Attributes**
- none

**Semantics**
- trigger host interrupt from the current QPU

**Verifier**
- none beyond basic op structure

**Side effects**
- writes `HostIRQ`

**Representation**
- direct hardware I/O write

**Expected lifetime**
- survives until bundle lowering

---

### 31. `vc4.thread_switch`

**Operands**: none  
**Results**: none

**Attributes**
- `mode` = `switch | last_switch`

**Semantics**
- cooperative hardware thread switch signal

**Verifier**
- legal only in functions with `threading = threadable`

**Side effects**
- writes thread scheduler state

**Representation**
- direct hardware signal in structured form

**Expected lifetime**
- lowers to a signaled bundle

---

### 32. `vc4.program_end`

**Operands**: none  
**Results**: none

**Attributes**
- none

**Semantics**
- end current QPU program / thread

**Verifier**
- later passes will enforce trailing restrictions; the first milestone only needs local structural legality

**Side effects**
- writes thread scheduler state

**Representation**
- direct hardware signal in structured form

**Expected lifetime**
- lowers to a signaled bundle

---

### 33. `vc4.async.wait`

**Operands**
- variadic `!vc4.async.token`

**Results**
- optional `!vc4.async.token` if the design chooses chaining; optional in the first milestone

**Attributes**
- optional `async` unit attr if chaining is supported

**Semantics**
- generic synchronization join for VC4 async activities (TMU/DMA primarily)

**Verifier**
- operands must all be `!vc4.async.token`

**Representation**
- structured compatibility op

**Expected lifetime**
- normalized to resource-specific waits later

---

## 10.7 Structured control-flow op

### 34. `vc4.cf.branch`

**Kind**: terminator

**Operands**
- no data operand in the hardware-facing design
- branch condition comes from flags, not an SSA `i1`

**Results**
- none

**Successors**
- one successor for unconditional
- two successors for conditional

**Attributes**
- `cond` : `VC4BranchCond`

**Semantics**
- branch based on current QPU flag state

**Verifier**
- `cond = always` requires one successor
- other conditions require two successors

**Representation**
- structured target control-flow concept, closely aligned with hardware flags

**Expected lifetime**
- lowers to `vc4.qpu.branch`

---

## 10.8 Host/system control ops

### 35. `vc4.enqueue_qpu`

**Operands**
- optional uniforms base address
- optional uniforms length
- optional launch/count operands if the implementation chooses to surface them now

**Results**
- optional token/result

**Attributes**
- symbol ref to target `vc4.func`

**Semantics**
- queue a user program request through the V3D scheduler user-program request interface

**Verifier**
- referenced symbol must be a `vc4.func`
- referenced function must be a kernel-like entry point

**Side effects**
- writes `QPUScheduler`

**Representation**
- structured host/device submission op

**Expected lifetime**
- survives until launcher/runtime lowering

---

### 36. `vc4.reserve_qpu`

**Operands**
- optional reservation masks

**Results**
- none

**Attributes**
- enough fields to represent exclusion of user programs on selected QPUs

**Semantics**
- configure QPU reservation masks

**Verifier**
- masks must fit target config when statically known

**Side effects**
- writes `QPUScheduler`

**Representation**
- direct V3D scheduler control, structured

**Expected lifetime**
- survives until launcher/runtime lowering

---

### 37. `vc4.v3d.query`

**Operands**
- optional selector operands

**Results**
- one or more scalar results

**Attributes**
- `kind` with at least:
  - `ident`
  - `queue_status`
  - `perf_counter`
  - `interrupt_status`
  - `error_status`
  - `scratch`

**Semantics**
- read V3D system/MMIO visible state

**Verifier**
- result arity and types must match `kind`

**Side effects**
- reads `V3DSystem`

**Representation**
- direct MMIO query, structured

**Expected lifetime**
- survives until runtime lowering

---

### 38. `vc4.v3d.configure`

**Operands**
- variant payload operands

**Results**
- none

**Attributes**
- `kind` with at least:
  - `cache_control`
  - `interrupt_enable`
  - `interrupt_disable`
  - `perf_map`
  - `perf_clear`
  - `perf_enable`
  - `vpm_reservation`
  - `vpm_allocator`
  - `scratch`

**Semantics**
- write V3D system/MMIO configuration

**Verifier**
- payload schema must match `kind`

**Side effects**
- writes `V3DSystem`

**Representation**
- direct MMIO config, structured

**Expected lifetime**
- survives until runtime lowering

---

## 10.9 Scheduled QPU sink ops

These ops are mandatory in the milestone even though no codegen is implemented yet. They are the dialect’s low-level sink surface.

### 39. `vc4.qpu.bundle`

**Operands**: none (all fields are attrs)  
**Results**: none  
**Regions**: none

**Required attributes**
- `sig`
- `unpack`
- `pm`
- `pack`
- `cond_add`
- `cond_mul`
- `set_flags`
- `write_swap`
- `waddr_add`
- `waddr_mul`
- `op_add`
- `op_mul`
- `raddr_a`
- either `raddr_b` or `small_imm`
- `add_a`
- `add_b`
- `mul_a`
- `mul_b`

**Semantics**
- one actual ALU/small-immediate QPU instruction word

**Verifier**
- exactly one of `raddr_b` and `small_imm` is active
- pack/unpack legality matches `pm`
- no illegal duplicate write to same accumulator/I/O
- signal choice is compatible with the rest of the instruction
- source mux values are legal
- write addresses are encodable

**Representation**
- direct hardware encoding

**Expected lifetime**
- survives to final emission

**Implementation note**
- This op should use custom parser/printer, because generic assembly is too hostile for humans.

---

### 40. `vc4.qpu.branch`

**Operands**: none  
**Results**: none  
**Regions**: exactly one delay-slot region

**Attributes**
- `cond`
- `relative`
- `use_reg`
- `raddr_a`
- `immediate`
- `write_swap`
- `waddr_add`
- `waddr_mul`

**Semantics**
- one hardware branch instruction with three explicit delay-slot instructions

**Verifier**
- delay-slot region contains **exactly three** scheduled QPU ops
- only scheduled-form functions may contain this op
- contained ops must themselves be scheduled QPU ops

**Representation**
- direct hardware encoding

**Expected lifetime**
- survives to final emission

---

### 41. `vc4.qpu.ldi`

**Operands**: none  
**Results**: none  
**Regions**: none

**Attributes**
- `mode` = `splat32 | per_elem_i2 | per_elem_u2`
- payload bits
- `pm`
- `pack`
- `cond_add`
- `cond_mul`
- `set_flags`
- `write_swap`
- `waddr_add`
- `waddr_mul`

**Semantics**
- one hardware load-immediate instruction

**Verifier**
- payload matches mode
- destination fields are encodable

**Representation**
- direct hardware encoding

**Expected lifetime**
- survives to final emission

---

### 42. `vc4.qpu.sema`

**Operands**: none  
**Results**: none  
**Regions**: none

**Attributes**
- `id`
- `mode` = `acquire | release`
- plus the same upper-half fields as load-immediate-style hardware encoding where needed:
  - `pm`
  - `pack`
  - `cond_add`
  - `cond_mul`
  - `set_flags`
  - `write_swap`
  - `waddr_add`
  - `waddr_mul`

**Semantics**
- one hardware semaphore instruction word

**Verifier**
- semaphore id range
- destination encodability
- sink-level legality consistent with the hardware restriction that the instruction may stall

**Representation**
- direct hardware encoding

**Expected lifetime**
- survives to final emission

---

## 11. Deferred hazards and sequence checks

The following are real hardware restrictions but may be enforced in later legalization/scheduling work instead of local op verifiers:

- no regfile read from a physical regfile location written by the immediately previous instruction
- SFU `r4` hazard window
- TMU_NOSWAP placement distance
- program-end trailing restrictions
- vector-rotate hazards involving `r5` or recently written accumulators

The sink ops and docs should acknowledge these restrictions, but the first milestone does **not** need a full schedule hazard analysis pass.

## 12. Testing requirements

The implementation is not complete unless it includes:

1. parser/printer round-trip tests for all custom types and all ops
2. verifier-negative tests for illegal attr/type combinations
3. `vc4-opt` smoke tests that load the dialect and parse example IR
4. scheduled-form tests covering:
   - legal `vc4.qpu.bundle`
   - legal `vc4.qpu.ldi`
   - legal `vc4.qpu.sema`
   - legal `vc4.qpu.branch` with exactly 3 delay slots
   - illegal branch delay-slot counts
5. function-form tests showing structured and scheduled forms are not mixed
6. descriptor tests covering TMU/VPM/DMA setup combinations

## 13. Deliverable definition for this milestone

The first `vc4` milestone is complete when the repository contains:

- a buildable standalone MLIR out-of-tree project
- a `vc4` dialect library
- a `vc4-opt` tool registering the dialect
- all ops in section 10
- all custom types in section 6
- the required enums / attrs for those ops
- parser/printer coverage
- verifier coverage
- lit/FileCheck tests
- project docs kept in sync with the implemented dialect surface

What is **not** required for this milestone:

- lowering from `gpu`
- scheduling passes
- register allocation
- qasm emission
- runtime/launcher generation
- fragment/TLB/control-list/shader-state support
