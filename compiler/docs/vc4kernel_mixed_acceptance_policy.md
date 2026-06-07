# VC4Kernel Mixed Acceptance Policy

P8.5 changes final hardware acceptance from replaying every historical isolated
fixture to running a smaller mixed suite that exercises phase interactions. The
isolated fixtures remain in tree as phase-local proofs and triage tools; they
are not weakened, deleted, or removed from explicit phase prompts.

## Routine Acceptance

After P8.5, routine final acceptance uses:

- the mixed acceptance manifest coverage checker;
- the mixed acceptance hardware runner once the suite is locked;
- Surface v2 audits and support-matrix checks;
- lit coverage and `check-vc4`.

Routine final acceptance does not run every isolated historical hardware
fixture by default. A phase prompt may still request isolated fixtures, and
triage should run the directly related isolated band when a mixed fixture fails.

## Manifest And Runner

The mixed suite is described by:

`compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_acceptance_manifest.json`

The manifest lists the required mixed fixtures, required P1-P9 feature coverage,
deterministic-reject coverage, and future P10-P13 extension points. Planned mode
allows fixture files to be absent while the suite is being added. Lock mode
requires every fixture input and expected result file to exist and verifies that
VC4Kernel mixed fixtures do not contain source-authored SSAVC4, scheduled VC4,
or producer dialects.

`run_mixed_acceptance.sh` reads the manifest and runs implemented fixtures with
the existing strict candidate hardware runners. It supports `--list` and
repeated `--fixture` arguments for triage. It creates a fresh
`VC4_CODEGEN_STATE_ROOT` by default, writes logs under that root, and prints
every `VC4_TEST_RESULT` line.

## Failure Triage

When a mixed fixture fails:

1. Keep the mixed fixture and oracle intact.
2. Run the isolated fixture band for the failed feature group.
3. If isolated fixtures pass, build a smaller interaction ladder.
4. Fix the compiler, lower half, runtime, or fixture bug and rerun the mixed
   fixture.

Deterministic rejects remain lit or audit tests. They are not hardware fixtures.

## P9 And Future Phases

P9 adds `mixed_subword_vpm_pack_unpack_roundtrip_vc4kernel` and
`mixed_quantized_gemv_subword_vpm_vc4kernel` to cover pack/unpack and subword
VPM/VDR/VDW transport in the mixed suite.

P10 adds `mixed_sfu_activation_tmu_vdw_vc4kernel` and
`mixed_sfu_norm_reduce_vpm_vc4kernel` to cover explicit approximate SFU policy,
SFU r4 wait behavior, exact/default math rejects, TMU safe loads, VDR/VPM tile
input, finite f32 reductions, VDW preserve, and P9 subword side paths in the
mixed suite. P10 mixed fixtures combine SFU with earlier accepted features
rather than replacing the mixed suite with isolated SFU smokes.

P11 adds `mixed_dynamic_rotate_reduction_scan_vc4kernel` and
`mixed_shuffle_vpm_tile_swizzle_vc4kernel` to the lock-mode mixed suite. The
final rotate surface has a static amount attr form and a dynamic scalar amount
operand form, both modulo 16, with output lane `l` reads source lane
`(l + (amount & 15)) & 15`. Dynamic lowering uses r5 and explicit hazard
spacing. Rotate-derived shuffle means only whole-vector horizontal rotate by one
scalar amount; arbitrary shuffle, permutation, and `vector.shuffle` producer
semantics remain deterministic rejects. P11 mixed fixtures combine dynamic
rotate with TMU safe loads, VDR/VPM tile paths, finite reductions, P9 subword
side paths, P10 SFU side paths, and VDW preserve stores.

P10-P13 must add mixed coverage for new accepted features instead of restoring
the old every-isolated-fixture final matrix. Future mixed additions should keep
the same strict requirements: CPU oracle, sentinels, expected PASS metadata, no
fixture-specific compiler special cases, and real hardware execution.

P9-P13 must add mixed coverage remains the standing policy phrase for audit
compatibility; after P9, the practical next extension point is P10.
