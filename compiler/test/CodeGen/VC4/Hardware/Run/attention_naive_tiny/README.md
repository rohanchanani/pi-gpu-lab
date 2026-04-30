
attention_naive_tiny

Validates full tiny scaled dot-product attention for one head without materializing a large score matrix.

For row-major tensors:

Q has shape [q_len, d]

K has shape [k_len, d]

V has shape [k_len, d]

out has shape [q_len, d]

The semantic result is:

score[k] = scale * dot(Q[q], K[k])

p[k] = softmax(score)[k]

out[q, t] = sum_k p[k] * V[k, t]

This tiny first version keeps k_len <= 7 and d <= 16, matching a one-vector value dimension. The stable oracle checks exact result fields and a maximum absolute error tolerance of 0.05.

The public API exposes only semantic buffers and dimensions:

attention_naive_tiny_prepare

attention_naive_tiny_launch

attention_naive_tiny_shutdown

For q_len == 0, there are no logical outputs. For k_len == 0, this reference defines each logical output element as 0.0f; this keeps the output shape [q_len, d] while avoiding undefined softmax over an empty key set.
