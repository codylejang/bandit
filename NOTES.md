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

backward pass scheduling? humans learn during task so we need to backprop more often?

evolution vs. lifetime learning: the episodes are "evolution" (tuning the brain's wiring), 
while within-episode hidden state persistence is "lifetime learning" (a single human doing the task)

curr model:
1. Sees feedback (identity + reward) → LSTM hidden state updates
2. Next time that identity appears → scorer reads the context and gives it a  
higher/lower score                                                          
3. When the reward gap is large, the accumulated evidence is strong enough to 
clearly differentiate → 90% accuracy                                         
4. When the gap is small, the signal is noisy → near chance (just like humans)

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