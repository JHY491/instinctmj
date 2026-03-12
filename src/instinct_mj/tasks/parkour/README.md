# Parkour Task

Usage notes for the `InstinctMJ` parkour task family on `mjlab`.

## Basic Usage Guidelines

**Task IDs:**

- `Instinct-Parkour-Simple-G1-v0` (train, noise-only terrain, no vision/AMP/MoE)
- `Instinct-Parkour-Simple-G1-Play-v0` (play, same)
- `Instinct-Parkour-Target-G1-v0` (train, non-AMP)
- `Instinct-Parkour-Target-G1-Play-v0` (play, non-AMP)
- `Instinct-Parkour-Target-Amp-G1-v0` (train, AMP)
- `Instinct-Parkour-Target-Amp-G1-Play-v0` (play, AMP)

1. For AMP parkour, go to `config/g1/g1_parkour_target_amp_cfg.py` and first update the local parkour dataset root:

   ```python
   _PARKOUR_DATASET_DIR = os.path.expanduser("~/Xyk/Datasets/data&model/parkour_motion_reference")
   ```

   `AmassMotionCfg.path` uses `_PARKOUR_DATASET_DIR` directly. If your filtered motion list is not stored as `parkour_motion_without_run.yaml` under that directory, also update `filtered_motion_selection_filepath` in the same file.

1. Train the policy:

```bash
instinct-train Instinct-Parkour-Target-G1-v0
instinct-train Instinct-Parkour-Target-Amp-G1-v0
```

1. Play trained policy (`--load-run` must be provided; absolute path is recommended. To visualize an untrained policy, use `--agent random`):

```bash
instinct-play Instinct-Parkour-Target-G1-Play-v0 --load-run <run_name> --viewer viser
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name>
```

1. Export trained policy (`--load-run` must be provided, absolute path is recommended):

```bash
instinct-play Instinct-Parkour-Target-G1-Play-v0 --load-run <run_name> --export-onnx
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --export-onnx
```

1. Use the exported ONNX policy for play:

```bash
instinct-play Instinct-Parkour-Target-G1-Play-v0 --load-run <run_name> --use-onnx
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --use-onnx
```

## Common Options

- `--num-envs`: Number of parallel environments (default varies by task)
- `--load-run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
- `--export-onnx`: Export the trained model to ONNX format for onboard deployment during playing
- `--use-onnx`: Use the ONNX model for inference during playing
