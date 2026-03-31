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