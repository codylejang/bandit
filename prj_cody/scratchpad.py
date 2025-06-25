# cd C:\Users\Luis\Dropbox\DELCARMEN\prj_cody
# .venv\Scripts\activate
# python
# .venv\Scripts\activate

from data import simulate_delta_agent, plot_actionval_timeseries

reward_probs = [0.8, 0.3]
epsilon = 0.5
num_trials = 500
actions, rewards, action_vals = simulate_delta_agent(reward_probs, epsilon, num_trials)

plot_actionval_timeseries(actions, rewards, action_vals, reward_probs)