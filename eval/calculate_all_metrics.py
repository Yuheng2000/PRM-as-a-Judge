"""
Aggregate task-level metrics from one evaluation run directory.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_episode_progress(pred_path: Path) -> list[float] | None:
    """
    Load the progress sequence from one `pred_vllm.json` file.
    """
    try:
        data = json.loads(pred_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Error: File not found at {pred_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file at {pred_path}")
        return None

    progress_list = []
    for step in data:
        progress = step.get("progress", 0.0)
        if progress is None:
            progress = 0.0
        progress_list.append(float(progress))
    return progress_list


def calculate_metrics(pred_paths: list[Path]) -> dict[str, float] | None:
    """
    Aggregate metrics across multiple episodes for one model-task pair.
    """
    milestone_counts = {0.25: 0, 0.50: 0, 0.75: 0, 1.0: 0}
    max_progress_list = []
    ppe_list = []
    ppl_list = []
    cra_list = []
    stagnation_list = []
    delta = 0.005

    valid_episode_count = 0
    for pred_path in pred_paths:
        raw_progress = load_episode_progress(pred_path)
        if raw_progress is None:
            continue

        valid_episode_count += 1
        if not raw_progress:
            max_progress_list.append(0.0)
            ppe_list.append(0.0)
            ppl_list.append(0.0)
            cra_list.append(0.0)
            stagnation_list.append(0.0)
            continue

        ep_max_progress = max(raw_progress)
        max_progress_list.append(ep_max_progress)

        if ep_max_progress >= 0.25:
            milestone_counts[0.25] += 1
        if ep_max_progress >= 0.50:
            milestone_counts[0.50] += 1
        if ep_max_progress >= 0.75:
            milestone_counts[0.75] += 1
        if ep_max_progress >= 1.0:
            milestone_counts[1.0] += 1

        progress_series = [0.0] + raw_progress
        num_steps = len(raw_progress)
        progress_final = progress_series[-1]
        path_length = sum(
            abs(progress_series[t] - progress_series[t - 1])
            for t in range(1, len(progress_series))
        )

        if path_length > 1e-9:
            ppe = progress_final / path_length
        else:
            ppe = 0.0
        ppe_list.append(ppe)
        ppl_list.append(progress_final * ppe)

        regret_sum = 0.0
        current_max = 0.0
        for t in range(1, len(progress_series)):
            value = progress_series[t]
            current_max = max(current_max, value)
            regret_sum += max(0.0, current_max - value)
        cra_list.append(regret_sum / num_steps if num_steps > 0 else 0.0)

        stagnation_count = 0
        for t in range(1, len(progress_series)):
            if abs(progress_series[t] - progress_series[t - 1]) < delta:
                stagnation_count += 1
        stagnation_list.append(stagnation_count / num_steps if num_steps > 0 else 0.0)

    if valid_episode_count == 0:
        return None

    metrics = {}
    for milestone in [0.25, 0.50, 0.75, 1.0]:
        rate = round(milestone_counts[milestone] / valid_episode_count * 100, 2)
        if milestone == 1.0:
            metrics["SR"] = rate
        else:
            metrics[f"M{int(milestone * 100)}"] = rate

    metrics["MaxP"] = round(float(np.mean(max_progress_list)) * 100, 2)
    metrics["PPL"] = round(float(np.mean(ppl_list)) * 100, 2)
    metrics["CRA"] = round(float(np.mean(cra_list)) * 100, 2)
    metrics["Stag"] = round(float(np.mean(stagnation_list)) * 100, 2)
    return metrics


def collect_task_pred_paths(model_dir: Path) -> dict[str, list[Path]]:
    """
    Collect `pred_vllm.json` files for each task under one model directory.
    """
    task_to_pred_paths: dict[str, list[Path]] = {}
    for task_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        pred_paths = sorted(task_dir.glob("*/pred_vllm.json"))
        if pred_paths:
            task_to_pred_paths[task_dir.name] = pred_paths
    return task_to_pred_paths


def process_benchmark(benchmark_dir: Path) -> None:
    """
    Compute metrics for all models under one benchmark directory.
    """
    print(f"Processing benchmark: {benchmark_dir}")
    all_rows = []
    all_tasks = set()

    for model_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
        row = {"Model": model_dir.name}
        task_to_pred_paths = collect_task_pred_paths(model_dir)

        for task_name, pred_paths in task_to_pred_paths.items():
            metrics = calculate_metrics(pred_paths)
            if metrics is None:
                continue
            all_tasks.add(task_name)
            for key, value in metrics.items():
                row[f"{task_name}_{key}"] = value

        all_rows.append(row)

    if not all_rows:
        print(f"No data collected for benchmark: {benchmark_dir.name}")
        return

    df = pd.DataFrame(all_rows)
    ordered_columns = ["Model"]
    metric_keys = ["M25", "M50", "M75", "SR", "MaxP", "PPL", "CRA", "Stag"]

    for task_name in sorted(all_tasks):
        for key in metric_keys:
            column_name = f"{task_name}_{key}"
            if column_name in df.columns:
                ordered_columns.append(column_name)

    remaining_cols = [column for column in df.columns if column not in ordered_columns]
    ordered_columns.extend(remaining_cols)
    df = df[ordered_columns]

    output_file = benchmark_dir / "metrics_summary.csv"
    print(f"Saving results to {output_file}")
    df.to_csv(output_file, index=False)
    print(df)


def process_result_path(result_path: Path) -> None:
    """
    Process one run directory and export one CSV per benchmark.
    """
    if not result_path.exists() or not result_path.is_dir():
        raise FileNotFoundError(f"Result path not found: {result_path}")

    benchmark_dirs = [
        p
        for p in sorted(result_path.iterdir())
        if p.is_dir() and p.name != "__pycache__"
    ]

    if not benchmark_dirs:
        print(f"No benchmark directories found under: {result_path}")
        return

    for benchmark_dir in benchmark_dirs:
        process_benchmark(benchmark_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark metrics from one run directory."
    )
    parser.add_argument("result_path", help="Run result directory, e.g. eval/results/run_260315_212722")
    args = parser.parse_args()

    process_result_path(Path(args.result_path))


if __name__ == "__main__":
    main()
