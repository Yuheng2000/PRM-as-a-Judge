# Eval Toolkit (Clean Version)

This folder contains a **minimal, reproducible evaluation toolkit** for PRM-as-a-Judge style trajectory auditing.

## What Was Kept

- Core runner: `run_judge.py`
- Aggregation scripts:
  - `calculate_all_metrics.py`
  - `calculate_step_metrics_per_trajectory.py`
- Official inference dependency: `examples/inference.py`
- A small demo benchmark: `demo_cases`
  - 2 tasks
  - 1 model folder (`demo_mix`)
  - 2 episodes per task (three views each)

## What Was Removed

To keep the repository clean and lightweight, the following were removed:

- Large-scale benchmark videos (e.g., RoboTwin full set)
- Historical run outputs and temporary caches
- Machine-specific legacy scripts and absolute-path local test scripts

## Directory Layout

```text
eval/
├── README.md
├── run_judge.py
├── run_eval.sh
├── calculate_all_metrics.py
├── calculate_step_metrics_per_trajectory.py
├── examples/
│   ├── blank.png
│   └── inference.py
├── tasks/
│   └── demo_cases.json
├── goals/
│   └── demo_cases/
│       ├── arrange_flowers/*.jpg
│       └── set_the_plates/*.jpg
├── videos/
│   └── demo_cases/
│       ├── arrange_flowers/demo_mix/*.mp4
│       └── set_the_plates/demo_mix/*.mp4
└── results/
    └── .gitkeep
```

## Important Note About Demo Videos

The current demo mp4 files are **placeholders (0-byte)** kept only to preserve filename/structure conventions.
To run real evaluation, replace them with valid videos following the same naming format:

```text
episodeX_{high|left|right}_*.mp4
```

## Quick Start

### 1) Dry-run sample discovery (no model loading)

```bash
python eval/run_judge.py \
  --benchmark demo_cases \
  --videos-root eval/videos \
  --tasks-root eval/tasks \
  --goals-root eval/goals \
  --output-root eval/results \
  --dry-run
```

### 2) Run evaluation

```bash
GRM_PATH=/path/to/Robo-Dopamine-GRM-8B-Pro bash eval/run_eval.sh
```

Optional runtime overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TP_SIZE=2 \
BATCH_SIZE=4 \
FRAME_INTERVAL=10 \
EVAL_MODE=backward \
GRM_PATH=/path/to/model \
bash eval/run_eval.sh
```

## Metrics Post-processing

### Aggregate metrics for a run directory

```bash
python eval/calculate_all_metrics.py eval/results/run_YYMMDD_HHMMSS
```

### Per-trajectory step metrics

```bash
python eval/calculate_step_metrics_per_trajectory.py eval/results/run_YYMMDD_HHMMSS
```

## Bring Your Own Data

Use this structure:

```text
eval/videos/{benchmark}/{task}/{model}/episodeX_{high|left|right}_*.mp4
eval/tasks/{benchmark}.json
eval/goals/{benchmark}/{task}/<single_goal_image>
```

`run_judge.py` will auto-discover valid triplets (`high/left/right`) and generate a run manifest.
