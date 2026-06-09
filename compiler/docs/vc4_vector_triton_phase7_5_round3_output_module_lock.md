# VC4 Vector/Triton Phase 7.5 Round 3 Output Module Lock

PHASE75R3_STATIC_FOUNDATION_LOCKED=YES
OUTPUT_MODULE_CONSTRUCTION=YES
TT_FUNC_REMOVE_WORKAROUND=NO
OLD_TTIR_OPS_DESTROYED_OR_REPLACED=YES
ATOMIC_IMPORT_FAILURE_BEHAVIOR=YES
MULTI_FUNCTION_MODULE_POLICY_TESTED=YES
PARTIAL_FAILURE_NO_PARTIAL_OUTPUT=YES
REPEATED_IMPORTER_NO_CRASH=YES
OUTPUT_MODULE_LIFETIME_AUDIT=PASS
SEMANTIC_STRING_MATCHING_AUDIT=PASS
PYTHON_SEMANTIC_IMPORTER_RETIRED=YES
DEFAULT_BUILD_TRITON_FREE=YES
OPTIONAL_CPP_FRONTEND_LANE_HEALTHY=YES
CPP_IMPORTER_STATIC_REGRESSION=PASS
READY_FOR_PHASE75R3E_HARDWARE_AND_PHASE8_READINESS=YES
READY_FOR_TRITON=NO

## Scope

Round 3 locks the non-hardware TTIR frontend foundation after the output-module
construction work. It does not add new TTIR features and does not claim broad
Triton support.

## Output-Module Strategy

`TritonToVC4Value` now analyzes the parsed TTIR input module without mutating it,
builds value-layer functions in a clean scratch module, verifies that output
module, and commits it into the original module only after every source function
lowers successfully.

The old `tt.func` remove/unlink workaround is gone. Source TTIR operations are
destroyed or replaced only during final commit through standard MLIR ownership.

The output verifier rejects TTIR, backend, VC4Kernel, SSAVC4, and scheduled VC4
dialects at the value boundary. Successful importer output is limited to the
standard value-layer/core dialects.

## Failure Atomicity

Unsupported TTIR input does not commit partial value IR. The source-controlled
tests cover a supported Phase 7.5 V1 function followed by an unsupported
`tt.dot` function and require the output file to remain absent or empty.

Running the importer again on its own value output deterministically rejects
because no `tt.func` remains. It does not crash and does not expose hidden or
unlinked TTIR operations.

## Multi-Function Policy

Multiple independent Phase 7.5 V1 `tt.func`s are supported. They lower
atomically to value-layer `func.func`s in deterministic source order. If any
function is unsupported, no function output is committed.

Cross-function calls remain outside the Phase 7.5 V1 support claim.

## Audits

Static audits now cover:

- typed semantic adapters and no printed type/attr/op semantic matching;
- no argument-name semantic role heuristic;
- no accepted Python semantic TTIR importer path;
- no `Operation::remove` or unlink workaround;
- no direct TTIR-to-target-dialect emission;
- raw Triton dependency closure isolated behind `VC4TritonFrontendDeps`;
- default build remains Triton-free;
- `vc4-triton-opt` remains isolated to the optional Triton-enabled build lane.

## Static Regression

Round 3d static regression passed:

- default `compiler/build` tools and `check-vc4`;
- optional `compiler/build-triton-llvm` tools and `check-vc4`;
- `compiler/test/Conversion/TritonToVC4Value`;
- `compiler/test/CodeGen/Triton`;
- `compiler/test/TritonFrontend`;
- `compiler/test/Conversion/VC4ValueToVC4Kernel`;
- `compiler/test/CodeGen/VC4Value`;
- `compiler/test/CodeGen/VC4Kernel`;
- `compiler/test/ValueSurface`;
- `compiler/test/Dialect/VC4Value`.

Static end-to-end pipelines for vector add, saxpy/select, and i32 add/select
lowered from source-controlled TTIR through value IR, VC4Kernel, SSAVC4, and
scheduled VC4. Boundary scans passed at every layer.

## Cleared Blockers

All non-hardware frontend foundation blockers are cleared:

- no remove/unlink workaround;
- no partial output on import failure;
- no string-based semantic classification;
- no Python semantic importer accepted path;
- no direct target dialect emission;
- no raw Triton dependency leakage into the default build.

## Remaining Work

Final hardware regression is still required before Phase 8. Round 3d is a
static foundation lock only and did not run hardware.

Staged TTIR features remain reductions, dot/contract, gather/scatter, block
pointers, rank-2 tensors, axes 1/2, non-V1 block sizes/ranges, control flow,
math/SFU policy, and subword/f16/bf16/int8 memory forms.

`READY_FOR_TRITON=NO` remains true.
