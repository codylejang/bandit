Before: The policy head was Linear(lstm_hidden → 2) — it mapped the LSTM's
context vector directly to 2 logits, one per slot position. The problem: the  
LSTM output is a single vector, and the two output neurons have independent
weights. Even with identical embeddings in the input, the two logits will     
differ due to weight asymmetry. The network learns "slot 0 tends to produce
slightly higher logits" and argmax amplifies this into 100% bias. No amount of
entropy regularization, zero-init, or input shuffling can fix this because
the asymmetry is structural — the two output neurons are fundamentally
different parameters.

After: The policy head is a shared scorer — a small network Linear(lstm_hidden
+ emb_dim → hidden → 1) that takes the LSTM context concatenated with one
option's embedding and outputs a single scalar score. At decision time, we run
both options through this same network independently:    

score_left  = scorer([context, emb_left])   # same weights
score_right = scorer([context, emb_right])  # same weights                    
logits = [score_left, score_right]
                                                                            
Since both options pass through the exact same weights, the only thing that   
can make the scores differ is the embedding content. If two options had       
identical embeddings, they'd get identical scores — guaranteed. The network is
forced to base its preference entirely on what it knows about each option
(from the embedding + LSTM context encoding reward history), not on which slot
it appeared in.

This is also more human-like: a person evaluates each painting based on its   
identity and past experience, not based on which side of the screen it's on.