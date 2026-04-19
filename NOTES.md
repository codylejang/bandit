# BASIC NOTES

if the special update rule is used, human retains memory of prev identity -> est val
therefore, should hidden states be reset across blocks?

first block, 1 of the 3 stimuli from the previous block was replaced with a new, unused stimulus, 
so effectively 2 familiar stimuli carried over and 1 was swapped out
But the win probabilities did not persist with those carried-over stimuli.
The paper says the reward probabilities were “reset at the beginning of every block”
familiarity carries, but value does not
if they were told that the probabilities would be reset, then this is not an accurate gauge. 
for instance, I would just treat all 3 as novel at the start of every block if i knew that they were being reset.
hence, i presume that this serves as only a familiarity bias gauge

backward pass scheduling? humans learn during task so we need to backprop more often?

evolution vs. lifetime learning: the episodes are "evolution" (tuning the brain's wiring), 
while within-episode hidden state persistence is "lifetime learning" (a single human doing the task)

curr model:
Step 1 — Stimulus: The LSTM receives the identity embeddings of the two       
offered options (left + right) concatenated with a "stimulus" flag. No        
decision happens here. This lets the LSTM encode which options are on the     
table before being asked to choose. The hidden state updates but no loss is  
computed.                                                                    

Step 2 — Decision: Same pair embeddings + "decision" flag. Now we tap the LSTM
output to:
- Score each option through the shared scorer (context + individual option    
embedding → scalar per option → softmax → action)                             
- Read the value head's estimate                                             
- Predict expected reward via the auxiliary head                              
                                                                            
This is the only step that produces loss-relevant quantities (log-prob, value,
aux prediction). The action is sampled here.                                 
                                                                            
Step 3 — Feedback: The chosen identity embedding is placed in its actual slot 
(left or right), the other slot is zeroed, and the actual reward scalar fills
the reward position in the state vector. This tells the LSTM "this identity,  
in this position, produced this reward." The hidden state absorbs the outcome
— no loss is computed, but this is how the LSTM learns to associate identities
with reward histories across trials within a block.

Then the update cycle: Every update_freq trials (default 5), the buffers of   
log-probs, values, rewards, and aux predictions are used to compute the
combined loss (policy gradient + value MSE + entropy bonus + aux reward       
prediction MSE). After backward + optimizer step, the hidden state is detached
so gradients don't flow back through the previous window, and the buffers   
reset. The hidden state values carry forward though — the LSTM remembers what
it saw, it just can't be credited/blamed for earlier trials in the next
backward pass.

- Odd blocks (1, 3, 5...): hold out the novel stimulus (index 2, which is     
always the newly introduced one)                                              
- Even blocks (2, 4, 6...): hold out a familiar one (randomly chosen from     
indices 0 or 1)

look at q vals in human data

train on simulation, test on human trial bandit data and measure similarity to humans (in q and bandit selection)

per episode, test the model on human study bandits, select the one with the highest similarity to humans
- different metrics can be observed to gauge similarity:
- q val estimation or direct bandit accuracy percentage per trial

embedding randomization structural change:
RL_persist.py:                                                                
  - randomize_embeddings() — re-initializes embedding table with fresh N(0,1)  
  vectors                                                                       
  - Called at the start of every training episode, eval episode, and greedy
  probe episode                                                                 
  - Embeddings removed from optimizer — only LSTM/policy weights are learned    
  - Checkpoints only save/load policy network weights (no embeddings)           
                                                                                
  eval_human_similarity.py:                                                     
  - Calls randomize_embeddings() per human session (one session = one episode)  
  - Human stim IDs remapped to sequential 0,1,2... slots (arbitrary, just needs 
  consistency within session)                                                  
  - No reserved slot ranges needed — every session gets completely fresh random 
  embeddings

  is it chance?
  - might even hard for two humans to perform similarly to each other
  - compare humans together see if the composite score is the same
  - non repeating pairs