import triton
import triton.language as tl


@triton.jit
def mixed_ttir_attention_apply_v0_axes_mask_cf_f16_storage_b16_kernel(S, VT, O, AUDIT, K, LDS, LDV, LDO, scale, BLOCK_SIZE: tl.constexpr):
    q = tl.program_id(0)
    d = tl.program_id(1)
    group = tl.program_id(2)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < K
    scores_raw = tl.load(S + q * LDS + offs, mask=mask, other=0.0).to(tl.float32)
    scores = scores_raw * scale
    active_scores = tl.where(mask, scores, -80.0)
    if group == 0:
        m = tl.max(active_scores, axis=0)
        shifted = active_scores - m
        e = tl.exp(shifted)
        e = tl.where(mask, e, 0.0)
        denom = tl.sum(e, axis=0)
        probs = e / denom
        v = tl.load(VT + d * LDV + offs, mask=mask, other=0.0).to(tl.float32)
        weighted = probs * v
        acc = tl.sum(weighted, axis=0)
        tl.store(O + q * LDO + d, acc)
        tl.store(AUDIT + q, denom)
