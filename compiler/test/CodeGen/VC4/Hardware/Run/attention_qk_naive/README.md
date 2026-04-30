
attention_qk_naive

Validates a tiny attention score kernel that computes S = Q * K^T without softmax or value projection.

For row-major tensors:

Q has shape [q_len, d]

K has shape [k_len, d]

scores has shape [q_len, k_len]

The semantic result is:

S[q, k] = scale * sum_{t=0..d-1}(Q[q, t] * K[k, t])

This first version intentionally keeps k_len <= 16 and d <= 16, matching one VC4 QPU vector for a score row. The stable oracle checks exact result fields and a maximum absolute error tolerance of 0.001.

The public API exposes only semantic buffers and dimensions:

attention_qk_naive_prepare

attention_qk_naive_launch

attention_qk_naive_shutdown

The reference bundle is a semantic hardware-run scaffold for the QK score baseline. It uses one logical runtime allocation, repeated launches, deterministic inputs, and guard verification.
