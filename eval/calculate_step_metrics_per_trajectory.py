"""
Compute per-step trajectory metrics from evaluation outputs.

The script recursively scans a result root, finds directories that contain both
`pred_vllm.json` and `result_summary.json`, and writes a `step_metrics.csv`
file next to them. Each CSV row corresponds to one trajectory prefix.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_episode_progress(pred_path: Path) -> list[float] | None:
    """
    Load the progress sequence from one `pred_vllm.json` file.

    Args:
        pred_path: Path to `pred_vllm.json`.

    Returns:
        A time-ordered list of progress values, or `None` if the file is
        missing or invalid.
    """
    try:
        data = json.loads(pred_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Warning: File not found at {pred_path}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON file at {pred_path}")
        return None

    progress_list: list[float] = []
    for step in data:
        progress = step.get("progress", 0.0)
        if progress is None:
            progress = 0.0
        progress_list.append(float(progress))
    return progress_list


def load_episode_success(episode_dir: Path) -> bool:
    """
    Read the success flag for one trajectory directory.

    Lookup order:
    1. `result_summary.json`
    2. `run_params.json`

    Args:
        episode_dir: Trajectory directory.

    Returns:
        `True` if a success flag is found and set, else `False`.
    """
    for name in ("result_summary.json", "run_params.json"):
        path = episode_dir / name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return bool(data.get("success", False))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return False


def calculate_prefix_metrics(progress_prefix: list[float], delta: float = 0.005) -> dict[str, float]:
    """
    Compute metrics for a prefix trajectory.

    Args:
        progress_prefix: Prefix sequence such as `progress[: t + 1]`.
        delta: Stagnation threshold.

    Returns:
        Metric dictionary for the prefix.
    """
    if not progress_prefix:
        return {
            "M25": 0.0,
            "M50": 0.0,
            "M75": 0.0,
            "SR": 0.0,
            "MaxP": 0.0,
            "PPL": 0.0,
            "CRA": 0.0,
            "Stag": 0.0,
        }

    max_progress = max(progress_prefix)
    progress_series = [0.0] + progress_prefix
    num_steps = len(progress_prefix)
    progress_final = progress_series[-1]

    path_length = sum(
        abs(progress_series[t] - progress_series[t - 1])
        for t in range(1, len(progress_series))
    )
    ppe = progress_final / path_length if path_length > 1e-9 else 0.0
    ppl = progress_final * ppe

    regret_sum = 0.0
    current_max = 0.0
    for t in range(1, len(progress_series)):
        value = progress_series[t]
        current_max = max(current_max, value)
        regret_sum += max(0.0, current_max - value)
    cra = regret_sum / num_steps if num_steps > 0 else 0.0

    stagnation_count = 0
    for t in range(1, len(progress_series)):
        if abs(progress_series[t] - progress_series[t - 1]) < delta:
            stagnation_count += 1
    stag = stagnation_count / num_steps if num_steps > 0 else 0.0

    return {
        "M25": 100.0 if max_progress >= 0.25 else 0.0,
        "M50": 100.0 if max_progress >= 0.50 else 0.0,
        "M75": 100.0 if max_progress >= 0.75 else 0.0,
        "SR": 100.0 if max_progress >= 1.0 else 0.0,
        "MaxP": round(max_progress * 100.0, 2),
        "PPL": round(ppl * 100.0, 2),
        "CRA": round(cra * 100.0, 2),
        "Stag": round(stag * 100.0, 2),
    }


def build_step_rows_for_trajectory(trajectory_dir: Path) -> list[dict[str, float | int | str]]:
    """
    Expand one trajectory directory into per-step metric rows.

    Args:
        trajectory_dir: Directory containing `pred_vllm.json` and
            `result_summary.json`.

    Returns:
        One row per time step. Returns an empty list if the prediction file is
        invalid.
    """
    pred_path = trajectory_dir / "pred_vllm.json"
    progress_list = load_episode_progress(pred_path)
    if progress_list is None:
        return []

    episode_success = int(load_episode_success(trajectory_dir))
    rows: list[dict[str, float | int | str]] = []

    for step_idx, progress_value in enumerate(progress_list, start=1):
        prefix = progress_list[:step_idx]
        metrics = calculate_prefix_metrics(prefix)

        row: dict[str, float | int | str] = {
            "Step": step_idx,
            "StepProgress": round(float(progress_value), 6),
            "EpisodeSuccess": episode_success,
        }
        row.update(metrics)
        rows.append(row)

    return rows


def iter_trajectory_dirs(root: Path) -> list[Path]:
    """Recursively find directories containing both result JSON files."""
    found: list[Path] = []
    for pred in root.rglob("pred_vllm.json"):
        if not pred.is_file():
            continue
        d = pred.parent
        if (d / "result_summary.json").is_file():
            found.append(d)
    return sorted(set(found))


def process_root(root: Path, output_filename: str = "step_metrics.csv") -> list[Path]:
    """
    Process a result root and write one per-step CSV for each trajectory.

    Args:
        root: Result root directory.
        output_filename: Output CSV name for each trajectory directory.

    Returns:
        List of written CSV paths.
    """
    root = root.resolve()
    print(f"Processing root (recursive): {root}")
    written_files: list[Path] = []

    ordered_columns = [
        "Step",
        "StepProgress",
        "EpisodeSuccess",
        "M25",
        "M50",
        "M75",
        "SR",
        "MaxP",
        "PPL",
        "CRA",
        "Stag",
    ]

    for trajectory_dir in iter_trajectory_dirs(root):
        rows = build_step_rows_for_trajectory(trajectory_dir)
        if not rows:
            print(f"  skip (no valid progress): {trajectory_dir.relative_to(root)}")
            continue

        df = pd.DataFrame(rows)
        cols = [c for c in ordered_columns if c in df.columns]
        df = df[cols]

        out_path = trajectory_dir / output_filename
        df.to_csv(out_path, index=False)
        written_files.append(out_path)
        print(f"  {trajectory_dir.relative_to(root)} -> {output_filename} ({len(df)} steps)")

    if not written_files:
        raise RuntimeError(
            f"No valid step-level data found under: {root} "
            f"(need directories with both pred_vllm.json and result_summary.json)"
        )

    print(f"Wrote {len(written_files)} trajectory CSV(s)")
    return written_files


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Recursively compute per-step trajectory metrics and export CSV files."
    )
    parser.add_argument(
        "result_path",
        help="Result root directory.",
    )
    parser.add_argument(
        "--output_filename",
        default="step_metrics.csv",
        help="Output CSV filename for each trajectory directory.",
    )
    args = parser.parse_args()

    root = Path(args.result_path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Result path not found: {root}")

    process_root(root, output_filename=args.output_filename)


if __name__ == "__main__":
    main()
