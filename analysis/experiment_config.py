
# PATHS
PROBEROWS_CSV_PATH = "data/probe_rows_01.csv" # version 1 "cody_code/probe_rows_00.csv"
# PROBEROWS_CSV_PATH = "data/probe_rows_greedy_01.csv" # version 1 "cody_code/probe_rows_00.csv"


# EXPERIMENT CONSTANTS 
N_H_FB_NEURONS = 128
N_H_DEC_NEURONS = 128
N_H_STIM_NEURONS = 128
N_PH_PRE_DEC_NEURONS = 64
N_PH_POST_DEC_NEURONS = 64
N_VH_PRE_DEC_NEURONS = 64
N_VH_POST_DEC_NEURONS = 64

HIDDEN_LAYERS =["h_stim", "h_dec", "h_fb"] # note in greedy dataset, lstm_out_dec = h_dec , but this is corrected when loading df
VALUE_HEAD_LAYERS = ["vh_pre_dec", "vh_post_dec"]
POLICY_HEAD_LAYERS = ["ph_pre_dec", "ph_post_dec"]
LAYERS = HIDDEN_LAYERS + VALUE_HEAD_LAYERS + POLICY_HEAD_LAYERS

LAYER_SIZES = {
    "h_stim": N_H_STIM_NEURONS,
    "h_dec": N_H_DEC_NEURONS,
    "h_fb": N_H_FB_NEURONS,
    "ph_pre_dec": N_PH_PRE_DEC_NEURONS,
    "ph_post_dec": N_PH_POST_DEC_NEURONS,
    "vh_pre_dec": N_VH_PRE_DEC_NEURONS,
    "vh_post_dec": N_VH_POST_DEC_NEURONS,
}

BANDIT_FEATURES = ["dQ", "dUnc", "dNov"]
CHOICE_VARS = ["choice_side","logit_diff"]
TRIALS_PER_BLOCK = 15