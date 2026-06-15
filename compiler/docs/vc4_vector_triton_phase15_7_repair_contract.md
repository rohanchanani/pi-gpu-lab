PHASE15_7_REPAIR_CONTRACT=LOCKED
ACCEPTED_PHASE15_TTIR_SNAPSHOTS_DO_NOT_REQUIRE_SCALAR_TT_LOAD=YES
ACCEPTED_PHASE15_TTIR_SNAPSHOTS_DO_NOT_REQUIRE_SCALAR_MEMREF_LOAD=YES
SCALAR_TT_LOAD_STAGED=YES
SCALAR_TT_LOAD_NONZERO_OTHER_STAGED=YES
TTIR_EXACT_DEFAULT_NO_POLICY_REJECT_RECLASSIFIED=YES
VALUE_EXACT_DEFAULT_MATH_REJECT_REMAINS_REQUIRED=YES
TTIR_EXACT_VS_APPROX_REQUIRES_STRUCTURAL_TARGET_PROFILE_INPUT=YES
NO_NAME_PATH_MANIFEST_SEMANTICS=YES
READY_FOR_PHASE15_7R1_CONTROLLED_FIXTURE_REPAIR=YES
PHASE15_CONTROLLED_FIXTURES_REPAIRED_AFTER_15_7_FAILURE=YES
MIXED_ACCEPTED_SNAPSHOT_HAS_SCALAR_TT_LOAD=NO
EXACT_DEFAULT_TTIR_NEGATIVE_RECLASSIFIED=YES
READY_FOR_PHASE15_7R2_IMPORTER_STATIC_RELOCK=YES
READY_FOR_TRITON=NO

# Phase 15.7 Repair Contract

Phase 15.7 failed because the controlled TTIR fixture contract was too broad
for the locked value surface. The failure did not identify a lower-half bug and
did not run hardware.

Accepted Phase 15 TTIR snapshots should include:

- `tl.exp` as public natural exp under explicit approximate-SFU finite policy.
- Reciprocal/division as approximate reciprocal/division under explicit finite
  nonzero-denominator policy.
- `tl.max` as finite f32 max reduction.
- One-block stable softmax v0 using natural exp semantics.
- A mixed softmax fixture only when the mixed form contains already-locked
  memory, control-flow, reduction, f16-storage, and SFU/softmax features.

Accepted Phase 15 TTIR snapshots must not require scalar `tt.load` or scalar
value-layer `memref.load`. Scalar `tt.load` is staged, and scalar `tt.load` with
nonzero `other` is staged. The accepted mixed fixture may use a scalar kernel
argument or constexpr for a scale value; it must not use scalar global-memory
load.

Source-controlled staged scalar-load fixtures are useful as negative coverage,
but they must not be listed as accepted/lowerable.

The `ttir_exact_math_no_policy_reject_b16` snapshot is reclassified as a bad
TTIR negative. Its emitted TTIR is structurally the same public `math.exp` form
as accepted `tl.exp`. Without a structural target-profile input in TTIR,
rejecting it while accepting `ttir_sfu_exp_f32_b16` would require source path,
fixture name, manifest entry, or kernel-name semantics. Those semantics are
forbidden.

Value exact/default math rejection remains required. The value-to-VC4Kernel
boundary must keep rejecting exact/default math without an explicit
approximate-SFU policy. TTIR exact/default rejection can be reinstated only when
the TTIR input carries a real structural target-profile marker or another
non-name semantic difference that the importer can inspect through MLIR APIs.

Phase 15.7R1 must repair the controlled fixture set before Phase 15.7 is rerun.
It must not broaden value memory support, implement scalar `memref.load`, or
use names/paths/manifest entries for semantic classification.

## Phase 15.7R1 Fixture Repair

The accepted mixed fixture now uses a scalar kernel argument for scale instead
of a scalar global-memory load. The accepted mixed TTIR snapshot must not
contain scalar `tt.load`, scalar-mask `tt.load`, or nonzero-`other` scalar
`tt.load`.

`ttir_scalar_load_nonzero_other_reject_b16` is the staged scalar-load fixture.
It preserves the observed scalar `tt.load` with scalar mask and `other=1.0` as
negative evidence. It is not accepted/lowerable.

`ttir_exact_math_no_policy_reject_b16` is reclassified as
`RECLASSIFIED_NOT_STRUCTURAL_TTIR_NEGATIVE`; value exact/default math rejection
remains required.
