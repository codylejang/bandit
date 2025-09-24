% Meta-RL Modeling of Human Behavior Using Recurrent Neural Networks 
% 
% This project explores how recurrent neural networks (RNNs) can be trained
% to mimic human learning and decision-making in dynamic environments.
% Using a simple bandit task, we first simulate behavioral data using a
% delta-rule agent — a common model of reinforcement learning. This data
% serves as a stand-in for human action and reward sequences that might be
% collected in real experiments. We then train a vanilla RNN on these
% sequences, using past actions and rewards as inputs to predict future
% actions.
% 
% Once trained, the RNN acts as a "frozen" agent — it no longer updates its
% weights, but instead adapts to new environments through changes in its
% internal hidden state. In a new bandit task with different reward
% contingencies, the RNN generates actions trial-by-trial based solely on
% its evolving memory of prior outcomes. This setup models how a human
% might adapt to a novel environment using previously learned strategies,
% without explicit re-learning or rule-based computation.
% 
% Together, this framework demonstrates a basic meta-reinforcement learning
% (meta-RL) system: the RNN learns to learn — not by updating parameters,
% but by shaping its internal dynamics. It offers a promising approach for
% modeling human flexibility and adaptation in sequential decision-making
% tasks.

% ---------------------------------------------------------
% ---------------------------------------------------------
% ---------------------------------------------------------

% Phase 1: Train RNN on delta-rule agent behavior for meta-RL
clc; clear

% ---------------------------------------------------------
% Simulate behavior that could represent human trial data
% ---------------------------------------------------------
% This uses a delta-rule agent to generate example data, but in practice,
% this would be replaced with actual human behavioral sequences (actions and rewards).
% The goal is to train an RNN to learn from these sequences alone —
% without knowing the underlying rule (if any) the human used.

reward_probs = [0.8, 0.3]; % true reward probabilities for left/right
epsi = 0.5; % epsilon-greedy exploration rate
num_trials = 10^5;
[A, R, Q] = BanditLearn_DeltaRule(reward_probs, epsi, num_trials);
% A: (num_trials x 1) vector of actions taken (integers: 1 or 2 )
% R: (num_trials x 1) vector of rewards received (binary: 0 or 1)


% ---------------------------------------------------------
% Prepare input (X) and target (Y) sequences for training
% ---------------------------------------------------------
% Prepare training data
A = categorical(A);                      % convert actions to categorical (needed for classificationLayer)
Ah = onehotencode(A, 2);                 % one-hot encode actions → (num_trials x num_bandits)


% Inputs: for each t, input = [one-hot(action_{t-1}); reward_{t-1}]
Xmat = [Ah(1:end-1,:) R(1:end-1)]';      % shape: (3 x num_trials−1)
                                         % rows = 2 for one-hot + 1 reward = 3 features

% Targets: for each t, target = action_t
Yvec = A(2:end)';                        % shape: (1 x num_trials−1), categorical row vector

% Format as sequence for trainNetwork (cell array of one sequence just matlab's syntax) 
X = {Xmat};
Y = {Yvec};

% ---------------------------------------------------------
% Define simple vanilla RNN architecture 
% ---------------------------------------------------------
layers = [
    sequenceInputLayer(size(X{1},1)) % input size = 3 
    fullyConnectedLayer(10)                          % hidden units
    tanhLayer
    fullyConnectedLayer(numel(categories(A)))        % output size = num unique actions
    softmaxLayer
    classificationLayer];

% Training settings
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 1, ...            % full sequence per batch
    'Verbose', 0, ...
    'Plots', 'training-progress');

% ---------------------------------------------------------
% Train the network to predict the next action
% ---------------------------------------------------------

% Train RNN
net = trainNetwork(X, Y, layers, options);

% Save layers for inference in Phase 2
save('trainedMetaRL_layers.mat', 'layers');

% ---------------------------------------------------------
% Evaluate how well the trained RNN predicts actions
% ---------------------------------------------------------
YPred = classify(net, X);
accuracy = mean(YPred{1} == Y{1});      % proportion correct across time
disp(['Prediction accuracy: ' num2str(accuracy)]);

% Plot predictions vs ground truth
% close all
% figure;
% plot(categorical(YPred{1}), 'o-'); hold on;
% plot(categorical(Y{1}), 'x--');
% legend('Predicted Action', 'Actual Action');
% xlabel('Timestep'); ylabel('Action');
% title('Predicted vs Actual Actions');

% % Plot hidden state activations
% act = activations(net, X, 'tanh', 'OutputAs', 'channels');
% act = act{1};
% figure;
% imagesc(act); colormap(jet); colorbar;
% xlabel('Time Step'); ylabel('Hidden Unit');
% title('Hidden State Activations');

%%
clc
net = load('trainedMetaRL_layers.mat', 'layers');          % from Phase 1
% net = dlnetwork(net.layers);   %% dlnetwork
disp(net); 

load('trainedMetaRL_layers.mat', 'layers');          % from Phase 1
lgraph = layerGraph(layers(1:end-1));                % remove classification layer (not needed for prediction)

netDL = dlnetwork(lgraph);                           % convert to dlnetwork to allow manual step-by-step input



clc
n_layers = numel(netDL.Layers);
layer_names = string({netDL.Layers.Name});



outcomes = {"Right, No Reward",[0     1     0];...
     "Right, Reward",[0     1     1];...
     "Left, No Reward",[1     0     0];...
     "Left, Reward",[1     0     1]};


outcome_names = [outcomes{:,1}]';
n_outcomes = numel(outcome_names);
% Rows are outcomes
df_stim = cell2table(outcomes,'VariableNames',{'Outcome','Code'});

for ii = 1:numel(layer_names)
	layer_name = layer_names(ii);
	df_stim.(layer_name) = cell(height(df_stim),1);
	for jj = 1:height(df_stim)
		stim = df_stim.Code(jj,:);
		resp = minibatchpredict(netDL,stim,Outputs=layer_name);
		df_stim.(layer_name){jj} = resp;

	end
	df_stim.(layer_name) = vertcat(df_stim.(layer_name){:});
end


%%
close all;
for ii = 1:numel(layer_names)
	
	layer_name = layer_names(ii);
	
	ax = nexttile;
	r = df_stim(:,layer_name);
	r = r.Variables;
	imagesc(r);
    title(layer_name,'Interpreter','none')

	if layer_name == "sequenceinput"
		ylabel('Outcome')
		ax.YTick = [1:n_outcomes];
		ax.YTickLabel = outcome_names;
	end
end

%% Phase 2: Use trained RNN in a new bandit task (frozen weights)
clc;

% ---------------------------------------------------------
% Setup: define a new bandit environment for the RNN to adapt to
% ---------------------------------------------------------
reward_probs = [0.3, 0.8];   % new reward probabilities (not seen during training)

% ---------------------------------------------------------
% Load trained network and convert to dlnetwork for inference
% ---------------------------------------------------------
load('trainedMetaRL_layers.mat', 'layers');          % from Phase 1
lgraph = layerGraph(layers(1:end-1));                % remove classification layer (not needed for prediction)
netDL = dlnetwork(lgraph);                           % convert to dlnetwork to allow manual step-by-step input

% ---------------------------------------------------------
% Initialize agent behavior loop
% ---------------------------------------------------------
prev_action = 1;        % arbitrary initial action (can also be random)
prev_reward = 1;        % arbitrary initial reward

phase2_num_trials = 500;
A_test = zeros(phase2_num_trials, 1);    % actions chosen by RNN agent
R_test = zeros(phase2_num_trials, 1);    % rewards received by RNN agent

for t = 1:phase2_num_trials

    % -----------------------------------------------------
    % Build input for time t based on previous experience
    % -----------------------------------------------------
    % Input format: [onehot(prev_action); prev_reward]
    x = [prev_action == 1; prev_action == 2; prev_reward];  % shape: (3 x 1)
    
    % Convert input to a sequence format for predict()
    x_seq = dlarray(reshape(x, [], 1, 1), 'CTB');            % shape: [features x time x batch]

    % -----------------------------------------------------
    % Forward pass through the RNN
    % -----------------------------------------------------
    y_dl = predict(netDL, x_seq);               % raw output logits
    p = extractdata(softmax(y_dl));             % convert to action probabilities

    % -----------------------------------------------------
    % Choose next action and observe outcome
    % -----------------------------------------------------
    a = randsample([1, 2], 1, true, p);         % sample action from softmax
    r = rand < reward_probs(a);                 % get reward based on chosen action

    % -----------------------------------------------------
    % Log behavior
    % -----------------------------------------------------
    A_test(t) = a;
    R_test(t) = r;

    % Update memory for next trial
    prev_action = a;
    prev_reward = r;

    % Optional: display behavior
    disp(['Trial ' num2str(t) ' | Action: ' num2str(a) ' | Reward: ' num2str(r) ...
          ' | P = [' num2str(p(1)) ', ' num2str(p(2)) ']']);
end

% ---------------------------------------------------------
% Plot the RNN agent's behavior and received rewards
% ---------------------------------------------------------
figure;
subplot(2,1,1);
plot(A_test, 'ko-');
ylabel('Action (1=left, 2=right)');
title('Meta-RL Agent Behavior in New Task');

subplot(2,1,2);
plot(R_test, 'g*-');
ylabel('Reward');
xlabel('Trial');
title('Received Rewards');




%%
