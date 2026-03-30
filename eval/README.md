# Eval Toolkit (Clean Version)

This folder contains a **minimal, reproducible evaluation toolkit** for PRM-as-a-Judge style trajectory auditing.

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

## Quick Start

### 1) Dry-run sample discovery

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
