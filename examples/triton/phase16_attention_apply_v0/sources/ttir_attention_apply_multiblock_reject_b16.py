import triton
import triton.language as tl


@triton.jit
def ttir_attention_apply_multiblock_reject_b16_kernel(S, VT, O, K, LDS, LDV, LDO, scale, NUM_KBLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    q = tl.program_id(0)
    d = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_SIZE)
    acc = 0.0
    for kb in tl.range(0, NUM_KBLOCKS):
        offs = kb * BLOCK_SIZE + lanes
        mask = offs < K
        scores_raw = tl.load(S + q * LDS + offs, mask=mask, other=0.0)
        scores = scores_raw * scale
        active_scores = tl.where(mask, scores, -80.0)
        m = tl.max(active_scores, axis=0)
        shifted = active_scores - m
        e = tl.exp(shifted)
        e = tl.where(mask, e, 0.0)
        denom = tl.sum(e, axis=0)
        probs = e / denom
        v = tl.load(VT + d * LDV + offs, mask=mask, other=0.0)
        weighted = probs * v
        acc += tl.sum(weighted, axis=0)
    tl.store(O + q * LDO + d, acc)
