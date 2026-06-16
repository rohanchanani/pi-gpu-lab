import triton
import triton.language as tl


@triton.jit
def ttir_online_softmax_normalizer_f32_b16_kernel(
    S, O, K, LDS, BLOCK_SIZE: tl.constexpr
):
    q = tl.program_id(0)
    m = -80.0
    l = 0.0
    for start in tl.range(0, K, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        scores_raw = tl.load(S + q * LDS + offs, mask=mask, other=0.0)
        active_scores = tl.where(mask, scores_raw, -80.0)
        block_m = tl.max(active_scores, axis=0)
        m_new = tl.maximum(m, block_m)
        alpha = tl.exp(m - m_new)
        beta = tl.exp(block_m - m_new)
        shifted = active_scores - block_m
        e = tl.exp(shifted)
        e = tl.where(mask, e, 0.0)
        block_l = tl.sum(e, axis=0)
        l = l * alpha + block_l * beta
        m = m_new
    inv_l = 1.0 / l
    tl.store(O + q, inv_l)
