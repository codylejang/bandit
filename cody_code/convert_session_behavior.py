# convert sessionBehavior .mat files to per-subject csvs
# skips ecog files (matlab table format not readable by scipy)

import os
import numpy as np
import pandas as pd
import scipy.io

MAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data",
                       "aquinoshareddata", "shared", "behavior", "sessionBehavior")
OUT_DIR = os.path.join(os.path.dirname(__file__), "session_behavior_csvs")
os.makedirs(OUT_DIR, exist_ok=True)


def extract_trials(mat_path):
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    ts = mat["taskStruct"]

    # ecog files use a different format with matlab tables
    if not hasattr(ts, "sessions"):
        return None

    sub_id = str(ts.subID)
    sessions = ts.sessions
    if not hasattr(sessions, "__len__"):
        sessions = [sessions]

    rows = []
    for si, sess in enumerate(sessions):
        blocks = sess.blocks
        if not hasattr(blocks, "__len__"):
            blocks = [blocks]

        for bi, blk in enumerate(blocks):
            n_trials = int(blk.numTrials)

            # some blocks are missing stimIDs/isPoolStim
            has_pool_info = hasattr(blk, "stimIDs") and hasattr(blk, "isPoolStim")
            pool_pwin = blk.pWin if has_pool_info else None
            pool_stim_ids = blk.stimIDs if has_pool_info else None

            for ti in range(n_trials):
                # which two stims were shown (pool indices)
                shown_idx = np.where(blk.isTrialStim[ti])[0]
                if len(shown_idx) != 2:
                    continue

                left_idx, right_idx = int(shown_idx[0]), int(shown_idx[1])

                # stim ids and pwin from pool
                if has_pool_info:
                    left_stim_id = int(pool_stim_ids[left_idx]) if not np.isnan(pool_stim_ids[left_idx]) else -1
                    right_stim_id = int(pool_stim_ids[right_idx]) if not np.isnan(pool_stim_ids[right_idx]) else -1
                    left_pwin = float(pool_pwin[left_idx])
                    right_pwin = float(pool_pwin[right_idx])
                else:
                    left_stim_id, right_stim_id = left_idx, right_idx
                    left_pwin = float(blk.pWin[left_idx]) if left_idx < len(blk.pWin) else np.nan
                    right_pwin = float(blk.pWin[right_idx]) if right_idx < len(blk.pWin) else np.nan

                # selected stim
                sel_idx = np.where(blk.isSelected[ti])[0]
                if len(sel_idx) != 1:
                    continue
                sel_idx = int(sel_idx[0])

                reward = 1 if len(np.where(blk.isWin[ti])[0]) > 0 else 0
                choice_side = 0 if sel_idx == left_idx else 1

                chosen_stim_id = left_stim_id if choice_side == 0 else right_stim_id

                rows.append({
                    "sub_id": sub_id,
                    "session": si,
                    "block": bi,
                    "trial": ti,
                    "left_stim_id": left_stim_id,
                    "right_stim_id": right_stim_id,
                    "left_pwin": left_pwin,
                    "right_pwin": right_pwin,
                    "chosen_stim_id": chosen_stim_id,
                    "choice_side": choice_side,
                    "reward": reward,
                    "rt": float(blk.RT[ti]) if hasattr(blk.RT, "__len__") else float(blk.RT),
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    mat_files = sorted([f for f in os.listdir(MAT_DIR) if f.endswith(".mat")])
    print(f"found {len(mat_files)} session files")

    converted, skipped = 0, 0
    for fname in mat_files:
        df = extract_trials(os.path.join(MAT_DIR, fname))
        if df is None:
            print(f"  SKIP (ecog table format): {fname}")
            skipped += 1
            continue
        out_name = fname.replace(".mat", ".csv")
        df.to_csv(os.path.join(OUT_DIR, out_name), index=False)
        print(f"  {out_name}: {len(df)} trials")
        converted += 1

    print(f"done: {converted} converted, {skipped} skipped")
