
import os
from helpers import norm_list
import polars as pl
from experiment_config import PROBEROWS_CSV_PATH, N_H_DEC_NEURONS, LAYERS
import polars.selectors as cs

def pull_neuron_column(df, layer, neuron_id, other_cols) -> pl.DataFrame:
    col_neur = build_layer_column(layer, neuron_id)
    if col_neur not in df.columns:
        raise ValueError(f"Neuron column not found: {col_neur}")
    other_cols = norm_list(other_cols)
    df_neuron = df.select([col_neur, *other_cols])
    df_neuron = df_neuron.rename({col_neur: "Activity"})
    return df_neuron

def load_dataframe(episode="max", verbose=True) -> pl.DataFrame:
    assert os.path.exists(PROBEROWS_CSV_PATH), f"File not found: {PROBEROWS_CSV_PATH}"
    df = pl.read_csv(PROBEROWS_CSV_PATH)
    if verbose:
        print("Loaded DataFrame:")
        describe_df(df)
    
    
    if "episode_idx" in df.columns:
        ep_idx_var = "episode_idx"
        is_greedy_dataset = False
    elif "greedy_episode_idx" in df.columns:
        ep_idx_var = "greedy_episode_idx"
        is_greedy_dataset = True
    else:
        raise ValueError("No episode index column found in DataFrame.")

    if is_greedy_dataset:
        # rename every lstm_out_dec to h_dec to match non-greedy dataset
        df = df.rename({f"lstm_out_dec_{i}": f"h_dec_{i}" for i in range(N_H_DEC_NEURONS)})


    # Filter by episode if specified
    max_episode = df.select(pl.col(ep_idx_var).max()).item()


    if episode != None:
        if episode == "max":
            df = df.filter(pl.col(ep_idx_var) == max_episode)
            df = df.drop(ep_idx_var)

        elif isinstance(episode, int):
            df = df.filter(pl.col(ep_idx_var) == episode)
            df = df.drop(ep_idx_var)
        else:
            raise ValueError("Episode must be 'max' or an integer.")
        if verbose:
            print(f"Filtered DataFrame where {ep_idx_var} == {episode}|(max ep={max_episode})")
            describe_df(df)

    return df



# FX: HELPER FUNCTIONS 
def parse_neuron_id(neuron_names) -> list[int]:
    """Extract neuron IDs from neuron names. ALWAYS the chars after the last underscore."""
    neur_ids = [int(name.split("_")[-1]) for name in neuron_names]
    return neur_ids

def parse_layer_from_name(neuron_names) -> list[str]:
    # can be one of any LAYERS
    layers = []
    for name in neuron_names:
        for layer in LAYERS:
            if layer in name:
                layers.append(layer)
                break

    assert len(layers) == len(neuron_names), "Some neuron names did not match any known layer."

    return layers
    

def layer_prefix(layer: str) -> str:
    assert layer in LAYERS, f"Layer must be one of {LAYERS}"
    return f"{layer}_"

def get_layer_columns(df: pl.DataFrame, layer: str) -> list[str]:
    prefix = layer_prefix(layer)
    cols = df.select(cs.starts_with(prefix)).columns
    if not cols:
        raise ValueError(f"No columns found for layer '{layer}' with prefix '{prefix}'.")
    return cols

def build_layer_column(layer: str, neuron_id: int) -> str:
    return f"{layer}_{neuron_id}"


def describe_df(df) -> None:
    print("DataFrame Head:")
    print(df.head())
    print("\nDataFrame Description:")
    print(df.describe())
    print(f"\nDataFrame Shape: {df.shape}")
