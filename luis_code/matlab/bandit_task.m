clc; clear; 

% INIT
% Number of trials and options
num_trials = 300;  % Total trials
reward_probs = [0.8, 0.2]; % True reward probabilities for options
num_options = length(reward_probs);   % Two-armed bandit (left and right options)

% Subject choices (1 = left, 2 = right)
Choice_hist = zeros( num_trials, 1); 
Reward_hist = zeros( num_trials, 1);  



% BETA ESTIMATION METHOD
% Initialize EV, uncertainty, and novelty for each option
alpha = ones(1, num_options); % Prior for wins (+1)
beta = ones(1, num_options);  % Prior for losses (+1)

% Novelty is initialized to 1 for each stimulus and decreases over time
novelty = ones(1, num_options);
novelty_decay = 0.5; % Rate at which novelty decreases per trial

% Store values for tracking
ev_hist = zeros(num_trials, num_options);
uncertainty_hist = zeros(num_trials, num_options);

% Store beta distributions at different trials (for plitting)
betapars_hist = containers.Map;
trial_points = 1:300; % you can make this sparser 

% SAMPLE AVERAGE EMTHOD
n_times_chosen = zeros(1, num_options);
rewards_summed = zeros(1, num_options);
Q_ave_hist = nan(num_trials, num_options);


% SIMULATE RUNS
for t = 1:num_trials
    chosen = randi([1, num_options]);  % The chosen option (1 or 2)
    reward = rand < reward_probs(chosen);  % Reward outcome (1 = win, 0 = loss)
	
	Choice_hist(t) = chosen; 
    Reward_hist(t) = reward;

	% Update EV and Uncertainty for the chosen option
	if reward
		alpha(chosen) = alpha(chosen) + 1;  % Increase wins
	else
		beta(chosen) = beta(chosen) + 1;    % Increase losses
	end
	[q_beta,beta_uncertainty] = betastat(alpha,beta);


    % Store values	
    ev_hist(t, :) = q_beta;
    uncertainty_hist(t, :) = beta_uncertainty;
	betapars_hist(num2str(t)) = [alpha; beta];
end

t
%%
% Plot Beta Distributions at Selected Trials
close all
clc
figure;
x = linspace(0, 1, 100);

y = cell(1,2);

for i = 1:2:length(trial_points)
    trial = trial_points(i);
    % subplot(1, length(trial_points), i);
    
    for opt = 1:num_options
        params = betapars_hist(num2str(trial));
        a = params(1, opt);
        b = params(2, opt);
        y{opt} = vertcat( y{opt} , betapdf(x, a, b));
    end
	i
end


for ii = 1:2
	ax = nexttile();

	imagesc(y{ii}')
	ax.YDir = "normal";
	title(sprintf('Beta Density for option %d', ii))
	xlabel('Trial');
	ylabel('Expected Value');
end
% legend('Option 1', 'Option 2');
    %     plot(x, y, 'LineWidth', 2); hold on;

% Plot the evolution of EV, Uncertainty, and Novelty over trials

% figure;
% subplot(3,1,1);

nexttile;
plot(ev_hist);
title('Expected Value (EV) Over Trials');
legend('Option 1', 'Option 2');

nexttile;
plot(uncertainty_hist);
title('Uncertainty Over Trials');
legend('Option 1', 'Option 2');


