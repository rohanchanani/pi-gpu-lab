# VC4Kernel Dialect Strict Specification

This document is the final Surface v2 contract for the VC4Kernel compute-kernel target layer. VC4Kernel sits below producer value dialects and above SSAVC4:

```
standard value layer -> vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware
```

There is no direct `VC4KernelToVC4` pipeline. Verified VC4Kernel hardware fixtures contain VC4Kernel operations and MLIR target carrier types such as `vector<16xi32>` or `vector<16xf32>`; they do not contain producer dialect operations.

## Final Status Categories

Every active Surface v2 row is in one of these categories:

- `accepted_hardware_proven`: executable VC4Kernel semantics with verifier, conversion, lower-half coverage where required, isolated hardware proof, mixed hardware proof, and mixed-claim audit coverage.
- `deterministic_reject`: a stable verifier, conversion, or audit reject with a category: `hardware_forbidden`, `static_surface_policy`, `not_meaningful`, `unsupported_source_layer_inside_vc4kernel`, or `out_of_scope_non_compute_hardware`.
- `internal_only`: acceptance infrastructure, audit, manifest, matrix, resource, or artifact contract that does not add executable kernel semantics.
- `out_of_scope_non_compute_hardware`: VC4 hardware domains outside the compute-kernel target, such as fixed-function graphics, tile-buffer color/Z/stencil paths, texture filtering, cube maps, and varyings.
- `producer_layer_future_work`: work above VC4Kernel. Such work may lower into accepted VC4Kernel operations but is not a reserved VC4Kernel feature.

No hardware-backed VC4Kernel compute-kernel mode is reserved without proof or an explicit deterministic reject.

## Final Operation Surface

The accepted VC4Kernel operation families are:

- Kernel and ABI: `vc4kernel.kernel`, `vc4kernel.return`, program id, number of programs, warp identity, lane-range identity.
- Predicates: full, empty, tail-prefix, rectangular-row, logical and/or/not, any, and all.
- Fragment values: constants, bitcast, splat, select, integer and finite floating-point comparisons, and reductions.
- ALU: accepted add-pipe and mul-pipe opcodes listed in the support matrix, including integer arithmetic, logical operations, shifts, and finite f32 arithmetic where hardware proof exists.
- Scalar subset: scalar arithmetic, control, address, and resource calculations needed by accepted target operations.
- Memory: TMU safe-offset loads, VPM QPU reads and writes, VDR global-to-VPM loads, VDW VPM/register-fragment stores, and runtime resource metadata.
- Subword movement: accepted w32, packed w16/w8, laned where admitted, pack/unpack, and selector-bearing paths listed in the support matrix.
- Approximate math: explicit fastmath/approx SFU operations only.
- Rotate-derived movement: dynamic rotate and the accepted rotate-derived composite policy. Arbitrary permutation is rejected.
- Barriers and cooperation: semaphore-backed barriers, cooperative resource metadata, and runtime/libpi descriptors required by accepted fixtures.
- Artifact emission: manifest and generated C metadata required by the runtime/hardware harness.

Removed operation spellings and producer-layer dialect operations are rejected inside verified VC4Kernel.

## Memory And Coherency

VC4Kernel names the actual compute memory paths:

- TMU is the register load path used for accepted safe-offset inactive loads. Inactive lanes use an explicit safe offset and do not infer a safe address from hidden state.
- VPM is the shared row storage visible to QPU reads/writes, VDR loads, and VDW stores.
- VDR moves global memory into VPM. VDW moves VPM or register fragments to global memory.
- VPM allocation, pitch, row bounds, hidden staging, spill pressure, and cooperative metadata are compiler-managed and checked by verifier, conversion, audit, and hardware fixtures.
- TMU-to-VPM is not an accepted substitute for VDR/VPM/VDW tile movement.
- Hidden spill reloads do not use TMU as an untracked memory path.

Full, tail-prefix, and rectangular inactive store policies preserve inactive output locations through the accepted VDW paths. Arbitrary sparse VDW stores are deterministic rejects in the final VC4Kernel surface.

## Dynamic Coordinates And Selectors

Dynamic coordinates are scalar i32 operands with verifier-enforced range, alignment, shape, and mode constraints. Vector or lane-varying coordinate operands are rejected.

The coordinate fields are separate:

- Row coordinate.
- Word-X coordinate.
- Subword selector.

Dynamic subword selector means byte or halfword selection inside an accepted packed mode. It is not dynamic width, dynamic subword mode, dynamic orientation, or dynamic layout. Dynamic orientation, dynamic width, and dynamic subword attrs are static-surface-policy rejects.

Setup-field masks isolate VC4 hardware bitfields. They are not modulo semantics and do not widen legal runtime values.

### VPM QPU Read/Write

Accepted QPU VPM dynamic selector semantics are separately proven for packed and laned modes, and for horizontal and vertical modes. Horizontal w32 dynamic word-X is not meaningful and is rejected. Subword selector readback through QPU VPM operations is a QPU-normalized readback view, not raw VDR carrier placement.

### VDR

Accepted VDR selector semantics:

- w32 has no selector.
- w16 packed uses selector `H` in `[0, 1]` and encodes `MODEW = 2 + H`.
- w8 packed uses selector `B` in `[0, 3]` and encodes `MODEW = 4 + B`.

Raw w32 VPM carrier placement and QPU subword readback are different views and are documented as such in the support matrix proofs.

### VDW

VDW selector proof is separate from VDR proof. VDW uses its own setup composition and preserve-store constraints; VDR/VDW symmetry is not assumed. DMA laned modes are hardware-forbidden rejects. Sparse VDW stores remain deterministic rejects.

## Pack/Unpack

Pack/unpack operations use explicit modes and are covered by verifier, conversion, isolated hardware fixtures, mixed fixtures, and mixed claim entries. Subword selector fields never change the static element width or orientation of an operation.

## SFU Policy

SFU-backed math is accepted only through explicit approximate/fastmath operations. Exact/default math does not lower to SFU. NaN, infinity, denormal, exception flag, exact rounding, and signed-zero behavior are outside the approximate SFU contract unless an accepted operation states otherwise.

## Rotate And Shuffle Policy

Dynamic rotate is accepted where represented by the proven rotate operation and lower-half path. Rotate-derived broadcast/reduction composites are accepted through enumerated operations. Arbitrary shuffle, arbitrary permutation, and source-layer shuffle dialect operations inside VC4Kernel are deterministic rejects.

## Resource Metadata And Runtime

Resource metadata is computed from accepted operations and carried through SSAVC4, scheduled VC4, artifacts, generated C, libpi descriptors, and the hardware harness. Metadata includes VPM row usage, hidden staging, barrier and semaphore counts, uniform layout, QPU slot requirements, and artifact manifest fields. Old resource dictionary fields with no semantic meaning are not accepted.

## Mixed Fixture Claim Policy

Mixed hardware acceptance is a final gate. A mixed fixture may claim a feature through `saw_*`, `no_*`, or equivalent result metadata only when the claim is present in:

```
compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_fixture_claims.json
```

The claim entry records `CHECKED_OUTPUT`, `CHECKED_AUDIT`, or `PHASE_GUARD` evidence. The audit script rejects missing evidence, dead checked-output paths, overly broad "full" claims, and phase guards that are not part of the same targeted/final run.

## VC4Kernel Surface v2 Final Lock

Final acceptance requires:

- The support matrix contains only final status categories.
- Every accepted feature has verifier, conversion, hardware, mixed, and claim-contract proof links.
- Every deterministic reject has a reject category and verifier/audit proof.
- The P12 selector contract proves row, word-X, and subword selector fields separately for VPM QPU, VDR, VDW, roundtrip, runtime stride/pitch, double-buffered, and spill-pressure cases.
- The mixed fixture claim audit, mixed manifest checker, P12 audit, matrix checker, relevant lit tests, `vc4-opt`, `vc4-codegen`, and `check-vc4` pass.
- No VC4Kernel compute-kernel feature remains outside these accepted, internal, out-of-scope, producer-layer, or deterministic-reject categories.

## Audit Traceability Terms

TMU inactive-load proof uses `safe_offset`, `inactive_load = #vc4kernel.inactive_load<zero>`, and address calculation as request base + safe_offset. safe-address inference is rejected.

VDW inactive-store proof uses `inactive_store = #vc4kernel.inactive_store<preserve>`. VDW stores require explicit inactive_store<preserve> for full, empty, and tail/active-prefix cases and for VPM-backed rectangular stores. sparse VDW store masks are not supported in P8, and P9 subword movement does not change that store policy.

SFU proof names `vc4kernel.fragment_sfu`, `#vc4kernel.fp_math_policy<approx_sfu>`, base-2 exponential, and base-2 logarithm. source-level `math.*` lowering is future work above VC4Kernel. exact/default math remains a deterministic reject for SFU lowering. P10 mixed fixtures cover the accepted approximate path.

Rotate proof names `vc4kernel.fragment_rotate` with a dynamic scalar amount, modulo 16 behavior, and the lane rule: output lane `l` reads source lane `(l + (amount & 15)) & 15`. The lower-half path uses r5. rotate-derived shuffle is accepted only through enumerated composites. arbitrary shuffle and vector.shuffle are deterministic-reject cases.

Lane broadcast is not a separate VC4Kernel operation. The final contract rejects `fragment_broadcast_lane` and uses fragment_select payload-or-zero, fragment_bitcast, finite f32 reduction policy, explicit finite numeric f32 broadcast, and generic bit-preserving f32 broadcast only through the accepted composite. The canonical lowering for vector/value producer work lowers into accepted operations and does not claim NaN, Inf, or signed-zero IEEE semantics. P11 mixed fixtures cover this policy.

P12 mixed fixtures keep word-X and subword selector are separate fields, dynamic selector is not dynamic subword mode, setup-field isolation, not modulo semantics, and VDR and VDW are asymmetric. No hardware-backed dynamic coordinate or selector mode is outside final status categories in P12.
