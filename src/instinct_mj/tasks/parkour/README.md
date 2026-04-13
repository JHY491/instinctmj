# Parkour Task

Usage notes for the `InstinctMJ` parkour task family on `mjlab`.

## Basic Usage Guidelines

**Task IDs (Adam SP):**

- `Instinct-Parkour-Adam-SP-v0` (train, with AMP, depth vision encoder, no MoE)
- `Instinct-Parkour-Adam-SP-Play-v0` (play, same)

1. For AMP parkour, update the local parkour dataset root in the corresponding config file:

   - Adam SP: `config/adam_sp/adam_sp_parkour_cfg.py`

   ```python
   _PARKOUR_DATASET_DIR = os.path.expanduser("~/Xyk/Datasets/data&model/parkour_motion_reference")
   ```

   `AmassMotionCfg.path` uses `_PARKOUR_DATASET_DIR` directly. If your filtered motion list is not stored as `parkour_motion_without_run.yaml` under that directory, also update `filtered_motion_selection_filepath` in the same file.

2. Train the policy:

```bash
instinct-train Instinct-Parkour-Adam-SP-v0
```

3. Play trained policy (`--load-run` must be provided; absolute path is recommended. To visualize an untrained policy, use `--agent random`):

```bash
instinct-play Instinct-Parkour-Adam-SP-Play-v0 --load-run <run_name> --viewer viser
```

4. Export trained policy (`--load-run` must be provided, absolute path is recommended):

```bash
instinct-play Instinct-Parkour-Adam-SP-Play-v0 --load-run <run_name> --export-onnx
```

5. Use the exported ONNX policy for play:

```bash
instinct-play Instinct-Parkour-Adam-SP-Play-v0 --load-run <run_name> --use-onnx
```

## Common Options

- `--num-envs`: Number of parallel environments (default varies by task)
- `--load-run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
- `--export-onnx`: Export the trained model to ONNX format for onboard deployment during playing
- `--use-onnx`: Use the ONNX model for inference during playing
