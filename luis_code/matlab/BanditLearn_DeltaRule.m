function [A,R,Q] = BanditLearn_DeltaRule(reward_probs, epsi, num_trials)
	% Number of trials and options

	num_bandits = length(reward_probs);
	
	% Store values for tracking
	Q = zeros(num_trials, num_bandits);
	R = zeros(num_trials, 1);
	A = zeros(num_trials, 1);
	
	
	% SIMULATE
	q = zeros(1, num_bandits);
	n = zeros(1, num_bandits);
	for t = 1:num_trials
	
		if rand < (1-epsi)
			[~,a] = max(q); % exploit
		else
			a = randi([1, num_bandits]); % explore
		end
	
		r = rand < reward_probs(a);
	
		n(a) = n(a)+1;
		lr =  1./n(a);
		q(a) = q(a) + lr*(r-q(a));
	
		A(t,1) = a;
		Q(t,:) = q;
		R(t,1) = r;
	end

	
end
