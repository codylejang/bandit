# data.py
import numpy as np

def simulate_delta_agent(reward_probs, epsilon, num_trials):
    """
    Simulate an agent using a simple delta rule with epsilon-greedy policy.
    
    Parameters:
        reward_probs: list of floats, e.g., [0.8, 0.3] for two arms
        epsilon: exploration probability
        num_trials: number of trials to simulate

    Returns:
        actions: np.array of chosen actions (0 or 1)
        isRewarded: np.array of received rewards (0 or 1)
        actionval_estimates: np.array of estimated action values over time
    """
    n_bandits = len(reward_probs)
    action_vals_t = np.zeros(n_bandits)
    num_actions_t = np.zeros(n_bandits)

    actions = np.zeros(num_trials, dtype=int)
    isRewarded = np.zeros(num_trials, dtype=int)
    actionval_estimates = np.zeros((num_trials, n_bandits), dtype=float)

    for t in range(num_trials):
        if np.random.rand() < epsilon:
            action = np.random.randint(n_bandits)
        else:
            action = np.argmax(action_vals_t)

        reward = np.random.rand() < reward_probs[action]
        num_actions_t[action] += 1
        lr = 1 / num_actions_t[action]
        action_vals_t[action] += lr * (reward - action_vals_t[action])

        actions[t] = action
        isRewarded[t] = reward
        actionval_estimates[t] = action_vals_t

    return actions, isRewarded, actionval_estimates

def plot_actionval_timeseries(actions, rewards,action_vals,reward_probs):
    """
    Plot action value estimates over time.
    """
    import plotly.express as px
    import pandas as pd

    # Validate lengths
    if len(actions) != len(rewards) or len(actions) != action_vals.shape[0]:
        raise ValueError("Length of actions, rewards, and action value estimates must match.")
    
    if len(reward_probs) != action_vals.shape[1]:
        raise ValueError("Number of action value estimates must match number of reward probabilities.")
    
    # Create a DataFrame for Plotly
    num_trials = len(actions)
    print("Actions:", actions)
    print("Rewards:", rewards)
    print("Action Value Estimates:", action_vals)

    data = {
        'Trial': np.arange(num_trials),
        'Action 0 Value Estimate': action_vals[:, 0],
        'Action 1 Value Estimate': action_vals[:, 1],
        'Action Taken': actions,
        'Reward Received': rewards
    }

    df = pd.DataFrame(data)
    fig = px.line(df, x='Trial', y=['Action 0 Value Estimate', 'Action 1 Value Estimate'],
                title='Action Value Estimates Over Trials',
                labels={'value': 'Value Estimate', 'variable': 'Action'},
                color_discrete_sequence=['blue', 'orange'])
    # # Add a scatter plot for actions taken
    fig.add_scatter(x=df['Trial'], y=df['Action Taken'], mode='markers', name='Action Taken',
                    marker=dict(color='red', size=5), showlegend=False)
    # Add a scatter plot for rewards received
    fig.add_scatter(x=df['Trial'], y=df['Reward Received'], mode='markers', name='Reward Received',
                    marker=dict(color='green', size=5), showlegend=False)
    
    # Add a horizontal line for each action's reward probability
    for i, prob in enumerate(reward_probs):
        fig.add_hline(y=prob, line_dash="dash", line_color="gray", 
                      annotation_text=f"Reward Prob {i}", annotation_position="top right")

    # show the plot
    # fig.show()
    fig.write_html("plotly_plot_temp_deleteafter.html", auto_open=True)