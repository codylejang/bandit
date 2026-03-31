Before: The policy head was Linear(lstm_hidden → 2) — it mapped the LSTM's
context vector directly to 2 logits, one per slot position. The LSTM received
[emb_L || emb_R || state] as input, so identity information was encoded in
the LSTM's context vector. The problem was not that identities were ignored —
they were represented in the context — but that the final output layer had
separate weight rows for logit_0 vs logit_1. These two independent sets of
parameters could develop different biases, making one slot systematically
preferred regardless of which identity occupied it. The network learns "slot 0
tends to produce slightly higher logits" and argmax amplifies this into 100%
bias. No amount of entropy regularization, zero-init, or input shuffling can
fix this because the asymmetry is structural — the two output neurons are
fundamentally different parameters that can diverge independently of the
identity content flowing through the LSTM.

After: The policy head is a shared scorer — a small network Linear(lstm_hidden
+ emb_dim → hidden → 1) that takes the LSTM context concatenated with one
option's embedding and outputs a single scalar score. At decision time, we run
both options through this same network independently:

score_left  = scorer([context, emb_left])   # same weights
score_right = scorer([context, emb_right])  # same weights
logits = [score_left, score_right]

Since both options pass through the exact same weights, the only thing that
can differentiate scores is the embedding content. As a thought experiment:
if you hypothetically placed the same identity in both slots, the old
architecture would still produce different logits (different output weight
rows), while the shared scorer would produce identical scores (same weights
applied to the same embedding). This is the symmetry guarantee — score
differences can only arise from differences in identity, never from slot
position.

This is also more human-like: a person evaluates each painting based on its
identity and past experience, not based on which side of the screen it's on.