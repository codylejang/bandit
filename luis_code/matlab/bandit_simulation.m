% Number of trials and options
num_trials = 300;  % Total trials
num_options = 2;   % Two-armed bandit (left and right options)

% Initialize EV, uncertainty, and novelty for each option
alpha = ones(1, num_options); % Prior for wins (+1)
beta = ones(1, num_options);  % Prior for losses (+1)

% Novelty is initialized to 1 for each stimulus and decreases over time
novelty = ones(1, num_options);
novelty_decay = 0.5; % Rate at which novelty decreases per trial

% Store values for tracking
EV_history = zeros(num_trials, num_options);
Uncertainty_history = zeros(num_trials, num_options);
Novelty_history = zeros(num_trials, num_options);

% Simulated subject choices (1 = left, 2 = right)
choices = randi([1, 2], num_trials, 1); % Example random choices
choices(1:end-100) = 1;
% Simulated reward outcomes (1 = win, 0 = no win)
reward_probs = [0.8, 0.2]; % True reward probabilities for options
rewards = rand(num_trials, 1) < reward_probs(choices)';

% Store beta distributions at different trials
beta_distributions = containers.Map;
trial_points = 1:300;[50, 150, 300];
A = [];
B = [];
% Loop through trials
for t = 1:num_trials
    chosen = choices(t);  % The chosen option (1 or 2)
    reward = rewards(t);  % Reward outcome (1 = win, 0 = loss)
    
    % Update EV and Uncertainty for the chosen option
    if reward
        alpha(chosen) = alpha(chosen) + 1;  % Increase wins
    else
        beta(chosen) = beta(chosen) + 1;    % Increase losses
    end
    
    % Compute Expected Value (EV) as the mean of the Beta distribution
    EV = alpha ./ (alpha + beta);
    
    % Compute Uncertainty as the variance of the Beta distribution
    Uncertainty = (alpha .* beta) ./ ((alpha + beta).^2 .* (alpha + beta + 1));
    
    % Novelty decreases for both options (even if not chosen)
    % novelty = max(0, novelty - novelty_decay); % Ensure it doesn't go below 0
    novelty = novelty - novelty_decay; % Ensure it doesn't go below 0
    
    % Store values
    EV_history(t, :) = EV;
    Uncertainty_history(t, :) = Uncertainty;
    Novelty_history(t, :) = novelty;
    
    % Store beta distributions at key trials
	% A =  [A;alpha];
	% B = [B;beta];
    % if ismember(t, trial_points)
        beta_distributions(num2str(t)) = [alpha; beta];
    % end
end


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
        params = beta_distributions(num2str(trial));
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
plot(EV_history);
title('Expected Value (EV) Over Trials');
legend('Option 1', 'Option 2');

nexttile;
plot(Uncertainty_history);
title('Uncertainty Over Trials');
legend('Option 1', 'Option 2');

nexttile;
plot(Novelty_history);
title('Novelty Over Trials');
legend('Option 1', 'Option 2');
