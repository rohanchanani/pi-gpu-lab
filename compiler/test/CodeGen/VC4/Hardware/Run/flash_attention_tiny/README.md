
flash_attention_tiny

Validates the algorithmic structure of tiny streaming attention using online max/sum accumulation without materializing the score matrix.

For row-major tensors:

Q has shape [q_len, d]

K has shape [k_len, d]

V has shape [k_len, d]

out has shape [q_len, d]

The semantic result is equivalent to scaled dot-product attention:

score[k] = scale * dot(Q[q], K[k])

p[k] = softmax(score)[k]

out[q, t] = sum_k p[k] * V[k, t]

The reference launch implementation uses the online update form:

m_new = max(m, s)

l_new = l * exp(m - m_new) + exp(s - m_new)

acc_new = acc * (l * exp(m - m_new) / l_new) + V * (exp(s - m_new) / l_new)

This tiny first version keeps q_len <= 4, k_len <= 7, and d <= 16, matching one QPU vector for the output dimension. The stable oracle checks exact result fields and a maximum absolute error tolerance of 0.08.
