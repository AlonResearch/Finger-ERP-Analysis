from pathlib import Path

from fingers_erp import calc_mean_erp


def find_file(filename: str) -> Path:
    """Locate a file in common locations (project folder, parent, or *data* subfolders)."""

    script_dir = Path(__file__).resolve().parent
    search_roots = []
    for candidate in (script_dir, script_dir.parent, Path.cwd()):
        resolved = candidate.resolve()
        if resolved not in search_roots:
            search_roots.append(resolved)

    data_dirs = []
    for root in search_roots:
        data_dirs.extend(
            sub for sub in root.iterdir() if sub.is_dir() and "data" in sub.name.lower()
        )

    for root in [*search_roots, *data_dirs]:
        candidate = root / filename
        if candidate.exists():
            return candidate

    searched = [str(p) for p in [*search_roots, *data_dirs]]
    raise FileNotFoundError(f"Could not locate {filename!r} in: {searched}")


if __name__ == "__main__":
    trial_file = find_file("events_file_ordered.csv")
    ecog_file = find_file("brain_data_channel_one.csv")

    fingers_erp_mean = calc_mean_erp(trial_file, ecog_file)
    print(f"ERP matrix shape: {fingers_erp_mean.shape}")
