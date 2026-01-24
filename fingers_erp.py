from pathlib import Path
from collections import Counter
from typing import Dict, List
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calc_mean_erp(trial_points: str | Path, ecog_data: str | Path) -> np.ndarray:
    """
    Compute mean ERPs per finger using a 200 ms pre-start to 1000 ms post-start window.

    Parameters
    ----------
    trial_points: path to CSV with columns starting_point, peak_point, finger
    ecog_data: path to CSV with a single column of ECOG samples

    Returns
    -------
    np.ndarray
        Matrix shaped (5, 1201) ordered by fingers 1..5
    """

    trial_path = Path(trial_points)
    ecog_path = Path(ecog_data)

    trials = pd.read_csv(trial_path)
    expected_cols = ["starting_point", "peak_point", "finger"]
    missing_cols = [c for c in expected_cols if c not in trials.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in trial_points: {missing_cols}")

    trials = trials.astype({col: int for col in expected_cols})

    ecog_df = pd.read_csv(ecog_path, header=None)
    if ecog_df.shape[1] != 1:
        raise ValueError("ecog_data must have exactly one column of signal values")
    ecog_signal = ecog_df.iloc[:, 0].to_numpy(dtype=float)

    pre_ms = 200
    post_ms = 1000
    window_len = pre_ms + post_ms + 1
    time_axis = np.arange(-pre_ms, post_ms + 1)

    finger_windows: Dict[int, List[np.ndarray]] = {finger: [] for finger in range(1, 6)}
    qa_counter: Counter[str] = Counter()

    for _, row in trials.iterrows():
        start = int(row["starting_point"])
        peak = int(row["peak_point"])
        finger = int(row["finger"])

        if finger not in finger_windows:
            qa_counter["finger_out_of_range"] += 1
            continue
        if peak <= start:
            qa_counter["peak_not_after_start"] += 1

        unclipped_start = start - pre_ms
        unclipped_end = start + post_ms
        window_start = max(0, unclipped_start)
        window_end = min(len(ecog_signal) - 1, unclipped_end)
        if window_start != unclipped_start or window_end != unclipped_end:
            qa_counter["window_clipped"] += 1

        if not (window_start <= peak <= window_end):
            qa_counter["peak_outside_window"] += 1

        raw_window = ecog_signal[window_start : window_end + 1]
        padded_window = np.full(window_len, np.nan, dtype=float)
        insert_start = pre_ms - (start - window_start)
        insert_end = insert_start + len(raw_window)
        padded_window[insert_start:insert_end] = raw_window

        finger_windows[finger].append(padded_window)

    total_valid = sum(len(v) for v in finger_windows.values())
    if total_valid == 0:
        raise ValueError("No valid trials remain after QA checks")

    fingers_erp_mean = np.empty((5, window_len), dtype=float)
    empty_fingers = []

    for idx, finger in enumerate(range(1, 6)):
        trials_for_finger = finger_windows[finger]
        if not trials_for_finger:
            fingers_erp_mean[idx, :] = np.nan
            empty_fingers.append(finger)
            continue
        stacked = np.vstack(trials_for_finger)
        fingers_erp_mean[idx, :] = stacked.mean(axis=0)

    if qa_counter:
        warnings.warn(
            "QA notes: " + ", ".join(f"{k}={v}" for k, v in qa_counter.items())
        )
    if empty_fingers:
        warnings.warn(f"No valid trials for fingers: {empty_fingers}")

    plt.figure(figsize=(10, 6))
    for idx, finger in enumerate(range(1, 6)):
        plt.plot(time_axis, fingers_erp_mean[idx, :], label=f"Finger {finger}")
    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Time (ms relative to start)")
    plt.ylabel("Amplitude")
    plt.title("Mean ERP per finger")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.show()

    return fingers_erp_mean
