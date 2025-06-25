# Main
# WINDOWS: .venv\Scripts\activate
# pip install torch numpy matplotlib plotly pandas
# pip freeze > requirements.txt
# cd C:\Users\Luis\Dropbox\DELCARMEN\prj_cody
# python

from train import train_rnn_on_behavior
from test import run_rnn_in_new_environment

if __name__ == "__main__":
    # Phase 1: train RNN on delta-rule agent behavior
    trained_model = train_rnn_on_behavior()

    # Phase 2: run trained RNN in a new bandit environment (weights frozen)
    run_rnn_in_new_environment(trained_model)
