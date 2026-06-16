import triton
import triton.language as tl


@triton.jit
def ttir_online_attention_nontransposed_v_reject_b16_kernel(
    S, V, O, K, LDS, LDV, LDO, scale, BLOCK_SIZE: tl.constexpr
):
    q = tl.program_id(0)
    d = tl.program_id(1)
    m = -80.0
    l = 0.0
    acc = 0.0
    for start in tl.range(0, K, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        scores_raw = tl.load(S + q * LDS + offs, mask=mask, other=0.0)
        scores = scores_raw * scale
        active_scores = tl.where(mask, scores, -80.0)
        block_m = tl.max(active_scores, axis=0)
        m_new = tl.maximum(m, block_m)
        alpha = tl.exp(m - m_new)
        beta = tl.exp(block_m - m_new)
        shifted = active_scores - block_m
        e = tl.exp(shifted)
        e = tl.where(mask, e, 0.0)
        block_l = tl.sum(e, axis=0)
        v = tl.load(V + offs * LDV + d, mask=mask, other=0.0)
        block_acc = tl.sum(e * v, axis=0)
        l = l * alpha + block_l * beta
        acc = acc * alpha + block_acc * beta
        m = m_new
    out = acc / l
    tl.store(O + q * LDO + d, out)
