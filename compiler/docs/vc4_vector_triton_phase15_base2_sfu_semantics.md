PHASE15_BASE2_SFU_SEMANTICS_CONTRACT=LOCKED
TARGET_SFU_EXP_IS_EXP2=YES
TARGET_SFU_LOG_IS_LOG2=YES
TARGET_SFU_RSQRT_IS_RECIPROCAL_SQRT=YES
TARGET_SFU_RECIP_IS_RECIPROCAL=YES
VALUE_MATH_EXP_IS_NATURAL_EXP=YES
VALUE_MATH_LOG_IS_NATURAL_LOG=YES
VALUE_MATH_SQRT_IS_SQRT=YES
TTIR_TL_EXP_IS_NATURAL_EXP=YES
TTIR_TL_LOG_IS_NATURAL_LOG=YES
TTIR_TL_SQRT_IS_SQRT=YES
NATURAL_EXP_LOWERING=EXP2_X_LOG2E
NATURAL_LOG_LOWERING=LOG2_X_LN2
SQRT_LOWERING=RSQRT_TIMES_X_POSITIVE_FINITE_DOMAIN
LOG2E_CONSTANT_REQUIRED=YES
LN2_CONSTANT_REQUIRED=YES
APPROX_MATH_POLICY=EXPLICIT
EXACT_DEFAULT_MATH_REJECTED=YES
PHASE15_5_EXP2_ORACLE_IF_PRESENT_REQUIRES_REPAIR=YES
VALUE_NATURAL_EXP_STATIC=PASS
VALUE_NATURAL_LOG_STATIC=PASS
VALUE_SQRT_STATIC=PASS
SQRT_LOWERING=RSQRT_TIMES_X
SOFTMAX_USES_NATURAL_EXP=YES
READY_FOR_PHASE15B_1_VALUE_NATURAL_MATH_STATIC_REPAIR=YES
READY_FOR_PHASE15B_2_VALUE_HARDWARE_EXP_LOG_SQRT_REPROOF=YES
READY_FOR_TRITON=NO

# Phase 15B Base-2 SFU Semantic Contract

The target VC4 SFU exp and log modes are base-2 hardware modes:

- target SFU `exp` means exp2;
- target SFU `log` means log2;
- target SFU `rsqrt` means reciprocal square root;
- target SFU `recip` means reciprocal.

The current target enum spellings may remain `exp` and `log`. Those names are
target mode spellings and must not be interpreted as public natural math
semantics.

Public value-layer and TTIR math semantics are different:

- value `math.exp` and TTIR `tl.exp` are natural exp;
- value `math.log` and TTIR `tl.log` are natural log;
- value `math.sqrt` and TTIR `tl.sqrt` are sqrt.

Therefore public natural functions must not lower directly to the target
base-2 modes. Under explicit approximate-SFU policy and finite domain metadata,
the intended repaired lowering is:

```text
natural exp(x) = target_exp2(x * log2(e))
natural log(x) = target_log2(x) * ln(2)
sqrt(x)        = x * target_rsqrt(x), for positive finite x
```

`LOG2E_CONSTANT_REQUIRED=YES` and `LN2_CONSTANT_REQUIRED=YES` are part of the
contract. The constants must be explicit in the lowering or represented through
a documented equivalent target composite.

Exact/default math remains rejected unless a future phase implements and proves
an exact or semantic-preserving emulation path. Approximate SFU lowering always
requires explicit policy metadata such as `vc4value.math_policy = "approx_sfu"`
and a finite domain policy. No exact/default math support is claimed here.

## Phase 15.5 Reclassification

Phase 15.5 hardware fixtures remain valid proofs of the current target SFU path,
but the exp/softmax oracles use base-2 `exp2_integer_ref`. They are not final
proofs of public natural `math.exp` or TTIR `tl.exp` semantics.

The marker below is intentional and TODO-free; Phase15B_1/Phase15B_2 must repair
static lowering/oracles and then re-prove public natural math:

```text
PHASE15_5_EXP2_ORACLE_IF_PRESENT_REQUIRES_REPAIR=YES
```

## Phase 15B.1 Static Repair

Phase 15B.1 statically repairs value lowering for public natural math:

- `VALUE_NATURAL_EXP_STATIC=PASS`
- `VALUE_NATURAL_LOG_STATIC=PASS`
- `VALUE_SQRT_STATIC=PASS`
- `SOFTMAX_USES_NATURAL_EXP=YES`

The static proof checks that natural exp reaches target exp2 only after the
`LOG2E` multiply, natural log reaches target log2 only before the `LN2`
multiply, and sqrt reaches target rsqrt only as the positive-finite
`x * rsqrt(x)` composite.

After the static repair, these invariants remain locked:

- any doc or test equating public `math.exp` with the target exp2 mode is
  wrong;
- any doc or test equating public `math.log` with the target log2 mode is
  wrong;
- target SFU exp/log may be discussed as target exp2/log2 modes only;
- `READY_FOR_TRITON=NO` remains locked.
