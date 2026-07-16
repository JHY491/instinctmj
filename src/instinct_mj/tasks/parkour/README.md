# Parkour Task

Usage notes for the `InstinctMJ` parkour task family on `mjlab`.

## Basic Usage Guidelines

**Task IDs (Adam SP):**

- `Instinct-Parkour-Adam-SP-v0` (train, with AMP, depth vision encoder, no MoE)
- `Instinct-Parkour-Adam-SP-Play-v0` (play, same)

**Task IDs (Adam Lite foothold — recommended):**

- `Instinct-Parkour-Adam-Lite-Foothold-VelEst-DeepCNN-v0` / `...-Play-v0`  
  SSR-aligned deep CNN + velocity estimator + imagined foothold guidance（楼梯）
- `Instinct-Parkour-Adam-Lite-Foothold-Mixed-v0` / `...-Play-v0`  
  同上 DeepCNN/VelEst 栈 + MoE×4 actor，mixed 多地形

详见 [`config/adam_lite/README.md`](config/adam_lite/README.md)。快捷脚本：

```bash
./scripts/train_parkour_adam_lite_foothold_velest_deep.sh
./scripts/train_parkour_adam_lite_foothold_mixed.sh
./scripts/play_parkour_adam_lite_foothold_velest_deep.sh --load_run <ckpt_or_run>
./scripts/play_parkour_adam_lite_foothold_mixed.sh --load-run <run>
```

1. For AMP parkour, update the local parkour dataset root in the corresponding config file:

   - Adam SP: `config/adam_sp/adam_sp_parkour_cfg.py`
   - Adam Lite: `config/adam_lite/adam_lite_parkour_cfg.py`（默认 `src/instinct_mj/motions`）

   ```python
   # Adam SP example
   _PARKOUR_DATASET_DIR = os.path.expanduser("~/Xyk/Datasets/data&model/parkour_motion_reference")
   ```

   `AmassMotionCfg.path` uses `_PARKOUR_DATASET_DIR` directly. If your filtered motion list differs, also update `filtered_motion_selection_filepath` in the same file.

2. Train the policy:

```bash
instinct-train Instinct-Parkour-Adam-SP-v0
instinct-train Instinct-Parkour-Adam-Lite-Foothold-VelEst-DeepCNN-v0
instinct-train Instinct-Parkour-Adam-Lite-Foothold-Mixed-v0
```

3. Play trained policy (`--load-run` must be provided; absolute path is recommended. To visualize an untrained policy, use `--agent random`):

```bash
instinct-play Instinct-Parkour-Adam-SP-Play-v0 --load-run <run_name> --viewer viser
instinct-play Instinct-Parkour-Adam-Lite-Foothold-VelEst-DeepCNN-Play-v0 --load-run <run_name>
instinct-play Instinct-Parkour-Adam-Lite-Foothold-Mixed-Play-v0 --load-run <run_name> --viewer viser
```

4. Export trained policy (`--load-run` must be provided, absolute path is recommended):

```bash
instinct-play Instinct-Parkour-Adam-SP-Play-v0 --load-run <run_name> --export-onnx
instinct-play Instinct-Parkour-Adam-Lite-Foothold-Mixed-Play-v0 --load-run <run_name> --export-onnx
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
