# Finger ERP Analysis

This mini-project computes event-related potentials (ERPs) from ECOG data aligned to finger movement events. It extracts 1201-sample windows (200 ms pre-start, 1 ms at start, 1000 ms post-start) for each trial, averages per finger, and plots the mean ERP per finger.

## Files
- `fingers_erp.py`: Implements `calc_mean_erp` that loads trial metadata and ECOG signal, performs QA, averages ERPs per finger, and plots the results.
- `main.py`: Script entry point that calls `calc_mean_erp` with the provided data files and prints the resulting matrix shape.

## Data layout
Expected inputs (CSV):
- `events_file_ordered.csv`: columns `starting_point`, `peak_point`, `finger` (all integers); each row is one trial.
- `brain_data_channel_one.csv`: one column of ECOG samples (floats), row count aligned with the indices in `events_file_ordered.csv`.

Placement rules: the script will look for each CSV in this folder, its parent folder (repo root), or any subfolder whose name contains `data` (case-insensitive). Keep the original filenames.

## Environment
Python 3.10+ with the following packages: `pandas`, `numpy`, `matplotlib`. Install with:

```bash
pip install -r requirements.txt
```

Or directly:

```bash
pip install pandas numpy matplotlib
```

## Running
From the repository root:

```bash
python main.py
```

This will compute the per-finger mean ERPs, display the plot, and print the ERP matrix shape.
