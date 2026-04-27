# Propositions for improving model-human similarity

## Context

We are training an LSTM actor-critic to perform a multi-armed bandit task modeled after Aquino et al., where stimulus identities (paintings) persist across blocks with changing reward probabilities. The goal is to find the training checkpoint whose decision behavior most closely resembles human subjects (n=22). Identity embeddings are re-randomized each episode so the model cannot memorize specific stimuli — it must learn a general strategy, like humans encountering novel paintings.

Human-human baseline: NLL ~ 0.68, choice agreement ~ 0.59.

## Architecture evolution

### Prop 1–3: Baseline improvements (failed)

Increased training duration (300 → 1K episodes), added per-N-trial updates (`update_freq=5`), and added an auxiliary reward prediction loss (`aux_coef`). None moved the model above chance.

**Root cause (diagnosed via gradient norms + logit analysis):** The shared option scorer took `[lstm_context, option_embedding]` as input, creating an additive bypass. The policy head produced scores directly from static embedding vectors, ignoring LSTM context entirely. Head gradients were 10–100x larger than LSTM gradients — the heads learned shortcuts while the LSTM received no learning signal.

### Prop 4: Context-only head (failed)

Removed the embedding bypass so the policy head read ONLY from LSTM hidden state (`context → 2 logits`). This fixed the bypass but caused immediate slot-bias collapse (`left%: 0.00` or `1.00`). The two output weights had asymmetric initialization, and policy gradient amplified this bias before the LSTM could learn anything. The model collapsed into "always pick left" — a local minimum it couldn't escape.

**Key insight from this failure:** Value belongs to *identities*, not to *positions*. The same painting appears on different sides across trials. The architecture must enforce identity-attributed scoring.

### Prop 5: Bilinear identity retrieval (current architecture)

The LSTM hidden state acts as memory over identity-value associations. The policy head projects context to a key vector, then dot-products against each option's embedding:

```
score(option) = (W_policy · context) · embedding(option)
```

Shared weights `W_policy` for both options. Aux reward predictor uses the same bilinear form with its own projection `W_aux`.

**Why this fixes both prior failures:**
- *No bypass:* score is zero when `W_policy · context = 0`. No additive pathway from embedding to score — the head is structurally forced to use the LSTM.
- *No slot bias:* same projection applied to same context, dotted with different embeddings. Slot-randomization prevents any embedding from consistently favoring one side.
- *Identity-attributed value:* differentiating options requires `W_policy · context` to align with the rewarded identity's embedding direction. The LSTM must encode "identity X has been rewarding" in its hidden state for the head to retrieve it.

## Training configuration evolution

### Update frequency: per-N-trial → per-block (reverted)

Initially moved from per-block to `update_freq=5` (Prop 2) for more frequent weight updates. Reverted to `update_freq=15` (one update per block) after diagnosing that mid-block detach truncated the credit path needed for within-block value tracking.

**The tradeoff:** With re-randomized embeddings, the LSTM must learn "encode feedback into hidden state such that it's retrievable when the same identity returns." That requires gradient flow from a retrieval trial back to the feedback trial. With `update_freq=5` + detach, retrieval and feedback are on opposite sides of a boundary most of the time (holdout + re-pairing spreads reappearances across 3–7 trials). The gradient never connects the two halves of the algorithm.

**Note on within-block learning:** `update_freq=15` does NOT reduce within-block adaptation. Weight updates (training) and hidden state updates (working memory) are independent timescales. The hidden state evolves at every forward pass step — stim, decision, feedback — regardless of when we step the optimizer. Within-block "learning" in the behavioral sense is a hidden-state phenomenon. The weights encode the *algorithm* for doing that working-memory update; that algorithm is learned across many episodes.

### Auxiliary coefficient: 0.5 → 2.0

Bumped to pressure the LSTM harder to encode identity-conditional reward info. In isolation this didn't help — the aux head stayed flat at ~0.5 regardless of reward history. The chicken-and-egg problem: aux can't provide identity-specific gradient unless the hidden state already encodes identity info, and the hidden state won't encode it without gradient signal.

### Embedding dimensionality: 16 → 4 (breakthrough)

**Diagnosis:** After exhausting architectural fixes (bilinear), signal strength (aux_coef), and credit path length (update_freq), the model remained at chance across all configurations. The binding problem was the bottleneck: the bilinear form requires `W_policy · context` to align with the rewarded identity's embedding in `id_emb_dim`-dimensional space. With 16 dims, the projection has 2048 parameters — all needing to be tuned via weak meta-learning signal.

**Fix:** Reduced `id_emb_dim` from 16 to 4. With only 3 identities active per block, 4 dims provides sufficient distinguishability (even for the 200-stimulus pool — random vectors in 4D have ~0.5 std cosine similarity, adequate for discrimination). The projection drops to 512 parameters, and the LSTM needs to learn a rotation in 4-space rather than 16-space.

**Cross-episode leakage concern (addressed):** Lower dimensionality increases embedding collisions across episodes. However, embeddings are drawn from a symmetric distribution each episode — no direction is systematically correlated with high reward. Over many episodes, the expected reward for any fixed direction is the global mean. The protection against leakage is the re-randomization itself, not the dimensionality.

**Result:** First sustained above-chance accuracy (0.55 overall, 0.58 on easy trials). First upward reward trend in training. First structured entropy drops. The LSTM is beginning to engage with the binding problem. Policy entropy shows increasing downward spikes — the model is occasionally becoming decisive rather than permanently indifferent.

### Prop 6: RL²-style explicit (last_chosen, last_action, last_reward) inputs

**Diagnosis at 3K episodes (post-Prop 5):** Performance still chance (149 reward). Per-trial diagnostics revealed logits were embedding-static — id47's logit stayed at -0.21 across an entire block regardless of rewards. The bilinear head was using the LSTM context, but the LSTM context wasn't evolving with reward feedback. The model also had `shape_r = -0.702` against humans — *anti-correlated*, having learned a slot/embedding-geometry artifact orthogonal to actual value.

**Root cause:** The LSTM was being asked to *infer* which identity got rewarded from a positional encoding (chosen embedding placed in its actual side slot at feedback step). This conflated identity with side and provided too weak a signal for binding.

**Fix:** Adopted RL² convention. At every step, the LSTM input now includes:
- `left_emb, right_emb` (offered pair, as before)
- `last_chosen_emb` (embedding of most recently chosen identity, zero at block start) — new third id slot
- `last_action` scalar (0/1, side of most recent choice)
- `last_reward` scalar (0/1, most recent reward observed)

The `last_*` fields update *before* each feedback step so the LSTM sees the just-observed event there and throughout the next trial's stim/decision. Dropped the positional chosen-marker trick at feedback. Input dim went from `2·emb + 5` to `3·emb + 7` (19 with `id_emb_dim=4`).

**Result (post-1500ep run):** Architectural progress, no task progress.
- *Logit dynamics finally appear:* swings of ±2.7–4.3 within blocks, identity-conditioned. Pre-RL² they were stuck at ±0.2 the whole block.
- *shape_r flipped from -0.702 to +0.091* — model is now roughly orthogonal to humans rather than actively inverted.
- *Some real per-id reward tracking visible:* model correctly prefers id71 (history `[0,1,1]`) over id6 (history `[0,0,1]`) when paired.
- *But still chance reward (150) and chance accuracy (0.52, hard > easy backwards).* NLL got *worse* (0.71 → 0.76) — model is more confidently wrong about humans.

**New failure mode: entropy collapse.** With ±2.7 logits, softmax assigns ~94% to one option. The model commits early in a block and won't revise (id123 logit pinned at -2.7 for trials 9–14 with no new evidence). Greedy eval freezes whatever wrong policy was locked in.

### Prop 7: Entropy-coefficient bump (current)

**Fix attempt:** `entropy_coef: 0.003 → 0.02` to keep the policy exploratory longer and prevent premature lock-in. Added eval-time mean policy entropy logging (`H: x.xxx` per eval cycle) — should stay above ~0.4 for most of training; collapse below ~0.1 early would confirm the failure mode.

### Model selection: dual-checkpoint tracking

Track both best-performing (highest eval reward) and most human-similar (lowest NLL against human trial data) checkpoints during training, with episode numbers recorded. At the end, restore and evaluate both.

**Why this matters:** At near-chance performance, "best reward" is just the luckiest Bernoulli roll — selection on noise. Human similarity gives a meaningful signal even when raw performance is flat. The two checkpoints may diverge: a model good at the task isn't necessarily the one that behaves like humans.

**Caveat discovered:** NLL as selection criterion has a floor problem. A model outputting uniform 50/50 predictions gets NLL ≈ ln(2) = 0.693. When no model has learned to track reward history, the "best NLL" is simply the most uniform checkpoint — not genuinely human-like. A truly human-similar model would have NLL *below* 0.693.

## Current status

Canonical config: `id_emb_dim=4` bilinear architecture + RL² inputs + `aux_coef=2.0`, `update_freq=15`, `entropy_coef=0.02`, dual-checkpoint selection.

**Trajectory of failure modes** (each row is the dominant problem after the prior fix):
1. Pre-bilinear: embedding bypass / slot bias collapse
2. Post-bilinear, pre-dim4: binding-problem bottleneck (16-d projection too sparse)
3. Post-dim4 (3K): logits embedding-static, LSTM not integrating feedback, anti-correlated with humans
4. Post-RL² (1500): identity binding works; logits dynamic ±2.7–4.3; *but* entropy collapses, model commits prematurely
5. Post-entropy-bump: TBD

Reward / accuracy still at chance through all of (3)–(4). The architectural transitions are real (different failure modes), but task performance has not yet moved.

Next step: 1500-ep run with `entropy_coef=0.02` to test whether keeping exploration alive lets the model *refine* per-id estimates rather than locking in early.

## External review notes

An external review (citing Bakker's RL-LSTM paper) flagged `returns = reward_t` as the main problem — arguing for multi-step returns / TD(λ) for long-horizon credit assignment. This critique is partially valid but misframes our task: the bandit has *immediate* reward (not delayed), so the "long horizon" is about *information flow* (past feedback informs future decisions), not *reward causation* (past action causes future reward). Multi-step returns would help indirectly (crediting exploratory actions with downstream information value) but are not the primary bottleneck. The review also incorrectly claimed that seeds were not set and that the LR scheduler was unused — both are present in the code.

The review's `gamma`-unused observation is correct: `gamma=0.99` is defined but `returns = reward_t` ignores it. Worth revisiting if multi-step returns are added.
