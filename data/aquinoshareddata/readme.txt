Data files for Neurons in human pre-supplementary motor area encode key computations for value-based choice, Aquino et al., Nature Human Behavior (2023).


# Behavior
## allBehavior.m
-transParams: Free parameters for optimal model. Parameter keys (as in getLLE_wsls.m):
    % softmax beta
    smB         = fitData.transParams(1);  
    % 'learning rate' parameter. Determins decay rate of weighting given to previous observations
    rlP         = fitData.transParams(2);
    % novelty initialization intercept, and terminal value
    nI          = fitData.transParams(3);
    nT          = fitData.transParams(4);
    % novelty utility
    nUtilI      = fitData.transParams(5);
    nUtilT      = fitData.transParams(6);
    % uncertainty utility intercept, terminal, blending
    uI          = fitData.transParams(7);
    uT          = fitData.transParams(8);
    uB          = fitData.transParams(9);
    % familiarity gated uncertaity
    fGateI      = fitData.transParams(10);
    % weight given to response action stickiness
    wActRep     = fitData.transParams(11);
    % weight given to stimulus identity stickiness
    wStimRep    = fitData.transParams(12);
    % propensity to use the left (+1) or right (-1) hand
    hI          = fitData.transParams(13);
    % binary flag: use familirity gate?
    uGate       = fitData.transParams(14); 

- qVals: q-values for left and right stimuli in each trial
- qVals_all: stored q-values for all possible stimuli
- Q1-Q5: distributional expectation quantiles
- RPE: reward prediction error
- KL: information gain in each trial [deprecated]
- pOption: probability that each option will be chosen, according to model
- pChoice: probability of each choice, according to model
- uVal: raw uncertainty of left/right stimuli
- nVal: raw novelty value of left/right stimuli
- isNovel: indicator for whether stimuli are novel or not
- prevResp: previous response
- selectHist: history of selected stimuli
- rejectHist: history of rejected stimuli
- wN: novelty initiation bias fed into utility 
- wUtilN: weight of novelty as separate utility component
- wU: weight of uncertainty
- fGate: familiarity gating number (interaction uncertainty x novelty, see Cockburn et al. (2022))
- nUtil: utility of novelty
- uUtil: utility of uncertainty
- prevRespFlag [deprecated]
- wsls [deprecated]
- uActRep [deprecated]
- sampleVal [deprecated]
- stimUtil: stimulus utility
- negLLE: negative log likelihood of decision vector
- selectedVector: which stimulus is selected in each trial
- rejectedVector: which stimulus is rejected in each trial
- blockID: which block trial belongs to
- outcome: binary trial outcome
- select_qVals: q-value of selected option
- reject_qVals: q-value of rejected option
- select_Q1-Q5 [deprecated]
- selectHistory_block: history of selected counts in block
- winHistory_block: history of wins in block
- trialStimHistory_block: history of stimulus presentations in block

## fitResults.m
- Output from hierarchical Bayesian inference for model comparison analysis. See Piray et al. (2018) for details on variable meaning.
cbm.output.model_frequency/cbm.output.exceedance_prob indicate main model comparison results for all sessions
cbm.output.responsibility indicates within session results


## sessionBehavior/XX_Sub_ExpExpTask_eCog_XX.m
Raw task data for each individual session separately
dispStruct: Psychtoolbox screen presentation information
taskStruct: taskStruct.sessions{...}.blocks{...} contains individual trial information  


# Neural 
## PXXCS/brainArea.m
Vector of dimensions [nClusters x 4]
Clusters with zeros in columns 2-4 were not accepted and should be disregarded
For accepted rows, variable key is: [channelNr, number of cluster in channel, clusterID, brainArea]
brainArea key: % 1=Right hippocampus, 2=Left hippocampus, 3=right amygdala, 4=left amygdala, 5=right dACC, 6=left dACC, 7=right preSMA, 8=left preSMA; 9=left vmPFC, 10=right vmPFC,

## PXXCS/sessionData.m
Main file containing neural data
-trialStartTime: first stim onset for each trial, in Neuralynx timestamp times in microseconds
-trialEndTime: trial end timestamp for each trial, referred to trial start/converted to seconds
-trialResponseTime: timestamp for button press, referred to trial start/converted to seconds
-trialOutcomeTime: timestamp for feedback presentation, referred to trial start/converted to seconds

unitCell (each row is a neuron):
- unitInfo: brainArea information [channelNr, number of cluster in channel, clusterID, brainArea]
- unreferencedSpikes: timestamp of each individual detected spike
- trialReferencedSpikes: time series of spikes cut by trials and referenced to trial start, converted to seconds
- decisionReferencedSpikes: time series of spikes cut by trials and referenced to button press, converted to seconds
- outcomeReferencedSpikes: time series of spikes cut by trials and referenced to feedback onset, converted to seconds

