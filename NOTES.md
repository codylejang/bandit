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