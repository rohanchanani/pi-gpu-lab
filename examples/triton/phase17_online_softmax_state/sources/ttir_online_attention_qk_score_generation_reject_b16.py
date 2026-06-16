import triton
import triton.language as tl


@triton.jit
def ttir_online_attention_qk_score_generation_reject_b16_kernel(
    Q, KT, VT, O, K, LDQ, LDK, LDV, LDO, scale, BLOCK_SIZE: tl.constexpr
):
    q = tl.program_id(0)
    d = tl.program_id(1)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < K
    qv = tl.load(Q + q * LDQ + offs, mask=mask, other=0.0)
    kv = tl.load(KT + q * LDK + offs, mask=mask, other=0.0)
    generated_score = tl.sum(qv * kv, axis=0) * scale
    m = tl.maximum(-80.0, generated_score)
    e = tl.exp(generated_score - m)
    v = tl.load(VT + d * LDV + offs, mask=mask, other=0.0)
    acc = tl.sum(e * v, axis=0)
    tl.store(O + q * LDO + d, acc)
