#!/usr/bin/env python3
"""
Generic rollout evaluation runner for judge / PRM backends.

The script discovers multi-view rollout samples, loads task prompts and
optional goal images, runs the backend model, and saves both per-sample and
run-level summaries.

Examples:
    python run_judge.py
    python run_judge.py --benchmark RoboTwin --task-filter handover --model-filter ACT
    python run_judge.py --dry-run
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_VIDEOS_ROOT = SCRIPT_DIR / "videos"
DEFAULT_TASKS_ROOT = SCRIPT_DIR / "tasks"
DEFAULT_GOALS_ROOT = SCRIPT_DIR / "goals"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "PRM" / "Robo-Dopamine-GRM-8B-Pro"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results"
DEFAULT_GOAL_IMAGE = SCRIPT_DIR / "examples" / "blank.png"

VIDEO_NAME_RE = re.compile(
    r"^(?P<sample>.+?)_(?P<view>high|left|right)(?:_.+)?\.mp4$"
)
VIEW_TO_CAM = {
    "high": "cam_high",
    "left": "cam_left_wrist",
    "right": "cam_right_wrist",
}
_GRM_INFERENCE = None


def safe_name(text: str) -> str:
    """Convert arbitrary text to a filesystem-safe name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "sample"

def write_json(path: Path, data: object) -> None:
    """Write a JSON file and create parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: Dict[str, object]) -> None:
    """Append one JSON object to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Dict[str, object]:
    """Read a JSON file and return a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_grm_inference():
    """
    Lazily import the official `GRMInference` class.

    This avoids loading heavy model dependencies during sample discovery or
    dry runs.
    """
    global _GRM_INFERENCE
    if _GRM_INFERENCE is not None:
        return _GRM_INFERENCE

    saved_local_rank = os.environ.get("LOCAL_RANK")
    saved_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    from examples.inference import GRMInference  # noqa: E402

    if saved_local_rank is None:
        os.environ.pop("LOCAL_RANK", None)
    else:
        os.environ["LOCAL_RANK"] = saved_local_rank

    if saved_cuda_visible_devices is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved_cuda_visible_devices

    _GRM_INFERENCE = GRMInference
    return _GRM_INFERENCE


def summarize_prediction(results: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Build a lightweight summary from `pred_vllm.json`.
    """
    if not results:
        return {
            "num_steps": 0,
            "final_progress": 0.0,
            "max_progress": 0.0,
            "min_progress": 0.0,
            "final_hop": 0.0,
        }

    progresses = [float(item.get("progress", 0.0)) for item in results]
    hops = [float(item.get("hop", 0.0)) for item in results]
    return {
        "num_steps": len(results),
        "final_progress": progresses[-1],
        "max_progress": max(progresses),
        "min_progress": min(progresses),
        "final_hop": hops[-1],
    }


def move_artifacts_to_sample_root(pipeline_output_dir: Path, sample_root: Path) -> None:
    """
    Flatten the output layout produced by the official `run_pipeline()`.

    The official implementation creates an extra timestamped subdirectory under
    `out_root`. This helper moves the artifacts back to `sample_root`.
    """
    for child in pipeline_output_dir.iterdir():
        target = sample_root / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))

    if pipeline_output_dir.exists():
        pipeline_output_dir.rmdir()


def load_task_prompts(tasks_root: Path) -> Dict[str, Dict[str, str]]:
    """
    Load task prompts from `tasks/*.json`.

    Returns:
        {benchmark: {task: task_description}}
    """
    prompt_map: Dict[str, Dict[str, str]] = {}
    for json_path in sorted(tasks_root.glob("*.json")):
        raw_data = read_json(json_path)
        prompt_map[json_path.stem] = {str(k): str(v) for k, v in raw_data.items()}
    return prompt_map


def load_goal_images(goals_root: Path) -> Dict[str, Dict[str, Path]]:
    """
    Load optional goal images from `goals/{benchmark}/{task}`.

    If a task does not provide a goal image, the caller can fall back to a
    default placeholder image.
    """
    goal_map: Dict[str, Dict[str, Path]] = defaultdict(dict)

    if not goals_root.exists():
        return goal_map

    for benchmark_dir in sorted(p for p in goals_root.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
            images = sorted(
                [
                    p
                    for p in task_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                ],
                key=lambda p: p.name,
            )
            if len(images) == 1:
                goal_map[benchmark_dir.name][task_dir.name] = images[0]
            elif len(images) > 1:
                raise ValueError(f"Expected a single goal image in: {task_dir}")

    return goal_map


@dataclass
class SampleSpec:
    """
    Standard sample representation used by the evaluation pipeline.
    """
    benchmark: str
    task: str
    model: str
    sample_id: str
    task_prompt: str
    goal_image: str
    input_videos: Dict[str, str]
    source_files: List[str]
    metadata: Dict[str, object] = field(default_factory=dict)

    def output_dir(self, run_root: Path) -> Path:
        """Return the output directory for this sample in the current run."""
        return (
            run_root
            / safe_name(self.benchmark)
            / safe_name(self.model)
            / safe_name(self.task)
            / safe_name(self.sample_id)
        )

    def as_dict(self) -> Dict[str, object]:
        """Serialize the sample to a JSON-friendly dictionary."""
        return {
            "benchmark": self.benchmark,
            "task": self.task,
            "model": self.model,
            "sample_id": self.sample_id,
            "task_prompt": self.task_prompt,
            "goal_image": self.goal_image,
            "input_videos": self.input_videos,
            "source_files": self.source_files,
            "metadata": self.metadata,
        }


def iter_benchmarks(videos_root: Path) -> Iterable[Path]:
    """Yield benchmark directories under the videos root."""
    return sorted(p for p in videos_root.iterdir() if p.is_dir())


def discover_samples(
    videos_root: Path,
    task_prompts: Dict[str, Dict[str, str]],
    goal_images: Dict[str, Dict[str, Path]],
    fallback_goal_image: Path,
) -> List[SampleSpec]:
    """
    Discover all valid rollout samples under the videos root.

    A sample is considered valid when it contains one `high`, one `left`, and
    one `right` video sharing the same filename prefix.
    """
    samples: List[SampleSpec] = []

    for benchmark_dir in iter_benchmarks(videos_root):
        benchmark = benchmark_dir.name
        benchmark_prompts = task_prompts.get(benchmark, {})
        benchmark_goals = goal_images.get(benchmark, {})

        for task_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
            task_name = task_dir.name
            task_prompt = benchmark_prompts.get(task_name)
            if not task_prompt:
                print(
                    f"[WARN] Skip task without prompt: benchmark={benchmark} task={task_name}",
                    file=sys.stderr,
                )
                continue

            goal_image = benchmark_goals.get(task_name, fallback_goal_image)
            for model_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                grouped: Dict[str, Dict[str, Path]] = defaultdict(dict)

                for video_file in sorted(model_dir.glob("*.mp4")):
                    match = VIDEO_NAME_RE.match(video_file.name)
                    if not match:
                        continue
                    sample_key = match.group("sample")
                    view = match.group("view")
                    if view in grouped[sample_key]:
                        print(
                            f"[WARN] Duplicate view file detected, keep first one: "
                            f"benchmark={benchmark} task={task_name} model={model_dir.name} "
                            f"sample={sample_key} view={view} duplicate={video_file.name}",
                            file=sys.stderr,
                        )
                        continue
                    grouped[sample_key][view] = video_file

                for sample_key, view_map in sorted(grouped.items()):
                    required_views = {"high", "left", "right"}
                    if set(view_map.keys()) != required_views:
                        missing_views = sorted(required_views - set(view_map.keys()))
                        print(
                            f"[WARN] Skip incomplete sample: benchmark={benchmark} task={task_name} "
                            f"model={model_dir.name} sample={sample_key} missing={missing_views}",
                            file=sys.stderr,
                        )
                        continue

                    sample_id = sample_key
                    input_videos = {
                        VIEW_TO_CAM[view]: str(view_map[view])
                        for view in ("high", "left", "right")
                    }
                    source_files = [str(view_map[view]) for view in ("high", "left", "right")]
                    samples.append(
                        SampleSpec(
                            benchmark=benchmark,
                            task=task_name,
                            model=model_dir.name,
                            sample_id=sample_id,
                            task_prompt=task_prompt,
                            goal_image=str(goal_image),
                            input_videos=input_videos,
                            source_files=source_files,
                            metadata={"sample_key": sample_key},
                        )
                    )

    samples.sort(key=lambda x: (x.benchmark, x.task, x.model, x.sample_id))
    return samples


def filter_samples(
    samples: List[SampleSpec],
    benchmark_filters: Optional[List[str]],
    task_filters: Optional[List[str]],
    model_filters: Optional[List[str]],
    sample_filters: Optional[List[str]],
    limit: Optional[int],
) -> List[SampleSpec]:
    """Filter samples using the provided CLI selectors."""
    selected = samples

    if benchmark_filters:
        allowed = {item.lower() for item in benchmark_filters}
        selected = [sample for sample in selected if sample.benchmark.lower() in allowed]

    if task_filters:
        tokens = [token.lower() for token in task_filters]
        selected = [
            sample
            for sample in selected
            if any(token in f"{sample.task} {sample.task_prompt}".lower() for token in tokens)
        ]

    if model_filters:
        tokens = [token.lower() for token in model_filters]
        selected = [
            sample for sample in selected if any(token in sample.model.lower() for token in tokens)
        ]

    if sample_filters:
        tokens = [token.lower() for token in sample_filters]
        selected = [
            sample for sample in selected if any(token in sample.sample_id.lower() for token in tokens)
        ]

    if limit is not None:
        selected = selected[:limit]
    return selected


def process_sample(
    model: object,
    sample: SampleSpec,
    run_root: Path,
    frame_interval: int,
    batch_size: int,
    eval_mode: str,
    visualize: bool,
    keep_cache: bool,
    skip_existing: bool,
) -> Dict[str, object]:
    """
    Run the official pipeline on one sample and save a compact summary.
    """
    sample_root = sample.output_dir(run_root)
    sample_root.mkdir(parents=True, exist_ok=True)

    result_summary_path = sample_root / "result_summary.json"
    if skip_existing and result_summary_path.exists():
        return json.loads(result_summary_path.read_text(encoding="utf-8"))

    pipeline_output_dir = Path(
        model.run_pipeline(
            cam_high_path=sample.input_videos["cam_high"],
            cam_left_path=sample.input_videos["cam_left_wrist"],
            cam_right_path=sample.input_videos["cam_right_wrist"],
            out_root=str(sample_root),
            task=sample.task_prompt,
            frame_interval=frame_interval,
            batch_size=batch_size,
            goal_image=sample.goal_image,
            eval_mode=eval_mode,
            visualize=visualize,
        )
    )
    move_artifacts_to_sample_root(pipeline_output_dir, sample_root)

    pred_path = sample_root / "pred_vllm.json"
    predictions = read_json(pred_path) if pred_path.exists() else []
    vis_path = sample_root / "reward_vis.mp4"
    visualization_path = str(vis_path) if vis_path.exists() else None

    summary = summarize_prediction(predictions)
    result = {
        **sample.as_dict(),
        "status": "ok",
        "eval_mode": eval_mode,
        "frame_interval": frame_interval,
        "batch_size": batch_size,
        "output_dir": str(sample_root),
        "pred_path": str(pred_path),
        "visualization_path": visualization_path,
        "summary": summary,
    }
    write_json(result_summary_path, result)

    if not keep_cache:
        cache_root = sample_root / ".cache"
        sample_json_path = sample_root / "sample.json"
        if cache_root.exists():
            shutil.rmtree(cache_root)
        if sample_json_path.exists():
            sample_json_path.unlink()

    return result


def aggregate_results(records: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Aggregate run-level statistics.
    """
    benchmark_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "ok": 0, "error": 0})
    model_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "ok": 0, "error": 0})

    for record in records:
        benchmark_key = str(record.get("benchmark", "unknown"))
        model_key = f"{record.get('benchmark', 'unknown')}/{record.get('model', 'unknown')}"

        for stats in (benchmark_stats[benchmark_key], model_stats[model_key]):
            stats["total"] += 1
            if record.get("status") == "ok":
                stats["ok"] += 1
            else:
                stats["error"] += 1

    return {
        "total_samples": len(records),
        "processed_ok": sum(1 for item in records if item.get("status") == "ok"),
        "processed_error": sum(1 for item in records if item.get("status") != "ok"),
        "benchmarks": dict(benchmark_stats),
        "models": dict(model_stats),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.
    """
    parser = argparse.ArgumentParser(description="Run judge / PRM evaluation on rollout videos.")
    parser.add_argument("--videos-root", type=Path, default=DEFAULT_VIDEOS_ROOT)
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    parser.add_argument("--goals-root", type=Path, default=DEFAULT_GOALS_ROOT)
    parser.add_argument("--grm-path", "--prm-path", dest="grm_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--goal-fallback", type=Path, default=DEFAULT_GOAL_IMAGE)
    parser.add_argument("--benchmark", nargs="*", default=None, help="Run only the selected benchmarks.")
    parser.add_argument("--task-filter", nargs="*", default=None, help="Filter by task name or task prompt.")
    parser.add_argument("--model-filter", nargs="*", default=None, help="Filter by model name.")
    parser.add_argument("--sample-filter", nargs="*", default=None, help="Filter by sample id.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of selected samples.")
    parser.add_argument("--frame-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--eval-mode", choices=["incremental", "forward", "backward"], default="backward")
    parser.add_argument("--visualize", action="store_true", help="Export reward visualization videos.")
    parser.add_argument("--keep-cache", action="store_true", help="Keep frame caches and sample.json.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples that already have results.")
    parser.add_argument("--dry-run", action="store_true", help="Only export the discovery manifest.")
    return parser


def main() -> None:
    """
    Main entry point.
    """
    args = build_arg_parser().parse_args()

    if not args.videos_root.exists():
        raise FileNotFoundError(f"videos_root not found: {args.videos_root}")
    if not args.tasks_root.exists():
        raise FileNotFoundError(f"tasks_root not found: {args.tasks_root}")
    if not args.goal_fallback.exists():
        raise FileNotFoundError(f"goal_fallback not found: {args.goal_fallback}")

    task_prompts = load_task_prompts(args.tasks_root)
    goal_images = load_goal_images(args.goals_root)
    all_samples = discover_samples(
        videos_root=args.videos_root,
        task_prompts=task_prompts,
        goal_images=goal_images,
        fallback_goal_image=args.goal_fallback,
    )
    selected_samples = filter_samples(
        samples=all_samples,
        benchmark_filters=args.benchmark,
        task_filters=args.task_filter,
        model_filters=args.model_filter,
        sample_filters=args.sample_filter,
        limit=args.limit,
    )

    run_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_root = args.output_root / f"run_{run_timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    manifest_path = run_root / "discovery_manifest.json"
    write_json(manifest_path, [sample.as_dict() for sample in selected_samples])
    print(f"[INFO] Discovered {len(selected_samples)} samples. Manifest saved to {manifest_path}")

    run_params = {
        "run_timestamp": run_timestamp,
        "argv": sys.argv,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    write_json(run_root / "run_params.json", run_params)

    if args.dry_run:
        return

    if not selected_samples:
        print("[INFO] No samples selected, exiting.")
        return

    if not args.grm_path.exists():
        raise FileNotFoundError(f"grm_path not found: {args.grm_path}")

    print(f"[INFO] Loading model from {args.grm_path}")
    GRMInference = load_grm_inference()
    model = GRMInference(
        str(args.grm_path),
        tensor_parallel_size=args.tensor_parallel_size,
    )

    results_jsonl_path = run_root / "results.jsonl"
    if results_jsonl_path.exists():
        results_jsonl_path.unlink()

    records: List[Dict[str, object]] = []
    total = len(selected_samples)
    for index, sample in enumerate(selected_samples, start=1):
        print(f"[{index}/{total}] {sample.benchmark} / {sample.task} / {sample.model} / {sample.sample_id}")
        try:
            record = process_sample(
                model=model,
                sample=sample,
                run_root=run_root,
                frame_interval=args.frame_interval,
                batch_size=args.batch_size,
                eval_mode=args.eval_mode,
                visualize=args.visualize,
                keep_cache=args.keep_cache,
                skip_existing=args.skip_existing,
            )
        except Exception as exc:
            record = {
                **sample.as_dict(),
                "status": "error",
                "error": str(exc),
            }
            error_dir = sample.output_dir(run_root)
            error_dir.mkdir(parents=True, exist_ok=True)
            write_json(error_dir / "result_summary.json", record)

        records.append(record)
        append_jsonl(results_jsonl_path, record)

    write_json(run_root / "results.json", records)
    summary = aggregate_results(records)
    write_json(run_root / "run_summary.json", summary)

    print("[INFO] Run complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
