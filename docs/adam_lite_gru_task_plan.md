# Adam Lite Foothold GRU Task Plan

## Goal

Create an isolated Adam Lite parkour task that adds the SSR-style `depth CNN + proprio MLP + GRU` encoder path without changing the current foothold task behavior.

The new task should be a clean experiment branch of the current Adam Lite foothold setup:

- Keep the existing foothold task unchanged.
- Keep AMP style prior and imagined foothold guidance strengths unchanged (no CReF on foothold tasks).
- Add GRU only as the policy perception encoder for a new task id.
- Treat depth as the latest available image, not a depth history sequence.
- Treat proprioception history as the GRU time sequence.

The paper-aligned input interpretation is:

```text
policy input at time t:
  current/latest depth image I_t
  proprioception history o^p_{t-h+1:t}

encoder:
  CNN(I_t) -> depth embedding
  MLP(o^p_k) per proprio frame -> proprio embedding sequence
  repeat depth embedding over h steps
  GRU([proprio_embed_k, depth_embed]) over k = t-h+1 ... t
  fusion/head -> latent z_t

actor:
  current proprio o^p_t + optional velocity estimate + z_t -> action a_t
```

Do not implement a fully recurrent actor-critic in the first version. The first version should use a fixed observation-history GRU inside the encoder, so PPO can keep its current minibatch/update behavior.

## Non-Goals

- Do not modify the existing task ids:
  - `Instinct-Parkour-Adam-Lite-Foothold-v0`
  - `Instinct-Parkour-Adam-Lite-Foothold-Play-v0`
- Do not change the existing foothold runner config used by those task ids.
- Do not downgrade AMP or foothold reward coefficients to make GRU easier to train.
- Do not add a true rollout-hidden recurrent PPO path in this phase.
- Do not make depth a 10 Hz held stream. Real deployment is 60 Hz camera with 50 Hz policy; training depth at 50 Hz with small delay randomization is acceptable.
- Do not modify Adam_SP or G1 code unless an error proves it is required.

## New Task Shape

Add new task ids:

```text
Instinct-Parkour-Adam-Lite-Foothold-GRU-v0
Instinct-Parkour-Adam-Lite-Foothold-GRU-Play-v0
```

Add new experiment name:

```text
adam_lite_foothold_gru
```

The new env config should be created as a separate file and should call the existing foothold env factory to get a fresh config object, then mutate only the new returned config object. This keeps the current foothold task unchanged.

Suggested new file:

```text
src/instinct_mj/tasks/parkour/config/adam_lite/adam_lite_foothold_gru_parkour_cfg.py
```

Suggested functions:

```python
def instinct_adam_lite_foothold_gru_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = instinct_adam_lite_foothold_env_cfg(play=play)
    configure_gru_policy_observations(cfg)
    return cfg

def instinct_adam_lite_foothold_gru_final_cfg(*, play: bool = False):
    cfg = instinct_adam_lite_foothold_gru_env_cfg(play=play)
    if play:
        cfg.viewer = ...
    return cfg
```

If the existing foothold config already has a final cfg function that handles viewer setup, reuse its behavior by calling the env-level function where possible and applying only GRU observation changes afterward. Avoid editing the existing foothold config file except for imports that are strictly necessary and harmless.

## Observation Design

The new task needs two policy observation roles:

1. Baseline actor proprioception.
2. Temporal proprioception for the GRU encoder.

The simplest safe implementation is to keep the existing actor observation terms unchanged and add GRU-specific observation terms with distinct names. This makes the new task a minimal superset of the current foothold baseline: the actor still receives the same non-GRU inputs as before, and the GRU latent is added as an extra encoded feature. A stricter paper-style ablation that reduces actor proprioception to only `o^p_t` can be added later as a separate task if needed.

Suggested policy terms:

```text
base_ang_vel
projected_gravity
velocity_commands
joint_pos
joint_vel
actions
depth_image

gru_base_ang_vel
gru_projected_gravity
gru_velocity_commands
gru_joint_pos
gru_joint_vel
gru_actions
```

Current actor proprio terms:

```text
keep the same history_length and flatten_history_dim as the current foothold task
```

GRU proprio terms:

```text
history_length=8 or 10
flatten_history_dim=False
```

Use `h=8` if preserving the current Adam Lite history length is preferred. Use `h=10` only if explicitly matching the paper-style temporal window is more important than keeping the current task shape.

Depth term:

```text
num_output_frames=1
history_skip_frames=1
delayed_frame_ranges=(0, 1)
camera_update_interval_frames: do not set
```

Reasoning:

- The real camera is 60 Hz and policy is 50 Hz.
- The policy consumes the latest available frame.
- The true frame age is roughly 0 to 16.7 ms depending on phase.
- If the simulator produces images at 50 Hz, integer-frame delay `(0, 1)` is a slightly conservative 0 to 20 ms perturbation.
- Do not emulate 10 Hz by holding each depth frame for 5 policy steps.
- If working on a branch that already has a `camera_update_interval_frames` option, leave it unset or equivalent to no hold.

The critic can keep the current privileged observation setup. If the current foothold task critic uses depth, leave that unchanged only for the new task if it is part of the current baseline. If the current branch already uses `height_map_priv` as critic-only terrain input, keep that path unchanged.

## GRU Encoder Implementation in instinct_rl

This part likely belongs in the sibling `instinct_rl` repository. Do not edit `instinct_rl` unless the user explicitly approves cross-repo changes.

Suggested new module:

```text
/Users/yangrurui/Desktop/instinct_rl/instinct_rl/modules/depth_proprio_gru_encoder.py
```

Suggested class:

```python
class DepthProprioGruEncoder(nn.Module):
    def __init__(
        self,
        input_segments: dict[str, tuple],
        depth_component_names: list[str],
        proprio_component_names: list[str],
        depth_channels: list[int],
        depth_kernel_sizes: list[int],
        depth_strides: list[int],
        depth_paddings: list[int],
        depth_embedding_dim: int,
        proprio_hidden_sizes: list[int],
        proprio_embedding_dim: int,
        gru_hidden_dim: int,
        gru_num_layers: int,
        fusion_hidden_sizes: list[int],
        output_size: int,
        nonlinearity: str,
        takeout_input_components: bool,
    ):
        ...
```

Expected input shapes:

```text
depth_image: (1, H, W) or (H, W)
GRU proprio terms combined: (T, D)
```

Expected internal shapes:

```text
depth image -> reshape to (B, C, H, W)
depth CNN -> (B, 128)
proprio history -> (B, T, D)
per-frame proprio MLP -> (B, T, 128)
depth repeat -> (B, T, 128)
concat -> (B, T, 256)
GRU -> output hidden (B, 256)
fusion MLP -> latent (B, 48)
```

Suggested first-version dimensions:

```text
depth CNN channels: [32, 64, 128]
depth CNN kernel sizes: [8, 4, 3]
depth CNN strides: [4, 2, 2]
depth embedding dim: 128
proprio MLP hidden: [512, 256]
proprio embedding dim: 128
GRU hidden dim: 256
GRU layers: 1
fusion MLP hidden: [512, 256, 128]
latent output size: 48
activation: ELU
```

Use a new `ParallelLayer` block as the preferred first implementation. This keeps the existing `EncoderActorCriticMixin` and `ParallelLayer.output_segment` behavior intact, and the new block only needs to map selected observation components to one latent tensor.

Update:

```text
/Users/yangrurui/Desktop/instinct_rl/instinct_rl/modules/parallel_layer.py
/Users/yangrurui/Desktop/instinct_rl/instinct_rl/modules/__init__.py
```

`ParallelLayer.build_output_segment()` should create the latent segment named:

```text
parallel_latent_0_depth_proprio_gru
```

because the block key in `encoder_configs` should be `depth_proprio_gru`.

Critical requirement:

The encoder must preserve or reconstruct the GRU time dimension. Do not call `get_subobs_by_components(..., temporal=False)` for the GRU proprio input and then treat the flattened vector as a sequence. Use `temporal=True` or equivalent explicit reshape from each component shape.

## InstinctMJ Config Dataclass

Add a config dataclass in:

```text
src/instinct_mj/rl/module_cfg.py
```

Suggested name:

```python
@dataclass
class InstinctRlDepthProprioGruEncoderCfg:
    class_name: str = "DepthProprioGruEncoder"
    depth_component_names: list[str] = field(default_factory=lambda: ["depth_image"])
    proprio_component_names: list[str] = field(default_factory=lambda: [
        "gru_base_ang_vel",
        "gru_projected_gravity",
        "gru_velocity_commands",
        "gru_joint_pos",
        "gru_joint_vel",
        "gru_actions",
    ])
    depth_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    depth_kernel_sizes: list[int] = field(default_factory=lambda: [8, 4, 3])
    depth_strides: list[int] = field(default_factory=lambda: [4, 2, 2])
    depth_paddings: list[int] = field(default_factory=lambda: [0, 0, 0])
    depth_embedding_dim: int = 128
    proprio_hidden_sizes: list[int] = field(default_factory=lambda: [512, 256])
    proprio_embedding_dim: int = 128
    gru_hidden_dim: int = 256
    gru_num_layers: int = 1
    fusion_hidden_sizes: list[int] = field(default_factory=lambda: [512, 256, 128])
    output_size: int = 48
    nonlinearity: str = "ELU"
    takeout_input_components: bool = True
```

Export it from:

```text
src/instinct_mj/rl/__init__.py
```

## Adam Lite Runner Config

Add a new policy config in:

```text
src/instinct_mj/tasks/parkour/config/adam_lite/agents/instinct_rl_amp_cfg.py
```

Suggested name:

```python
class FootholdGruVisionPolicyCfg(InstinctRlEncoderActorCriticCfg):
    ...
```

Suggested settings:

```python
encoder_configs = {
    "depth_proprio_gru": InstinctRlDepthProprioGruEncoderCfg(),
}
critic_encoder_configs = existing current foothold behavior
```

If using `EncoderTerrainActorCriticCfg` for auxiliary height-map prediction, set:

```python
terrain_latent_component = "parallel_latent_0_depth_proprio_gru"
```

If using plain `InstinctRlEncoderActorCriticCfg`, the actor simply receives:

```text
  baseline actor proprio + parallel_latent_0_depth_proprio_gru
```

Add a new runner config:

```python
@dataclass(kw_only=True)
class AdamLiteFootholdGruVisionAmpRunnerCfg(AdamLiteFootholdVisionAmpRunnerCfg or base foothold runner):
    experiment_name: str = "adam_lite_foothold_gru"
    policy: InstinctRlActorCriticCfg = field(default_factory=lambda: FootholdGruVisionPolicyCfg())

def adam_lite_foothold_gru_amp_runner_cfg():
    return AdamLiteFootholdGruVisionAmpRunnerCfg()
```

Do not change the existing foothold runner factory.

## Task Registry

Update:

```text
src/instinct_mj/tasks/parkour/config/adam_lite/__init__.py
```

Add only new factories and registrations:

```python
def _foothold_gru_parkour_cfg(play: bool):
    from .adam_lite_foothold_gru_parkour_cfg import instinct_adam_lite_foothold_gru_final_cfg
    return instinct_adam_lite_foothold_gru_final_cfg(play=play)

def _foothold_gru_runner_cfg():
    from .agents.instinct_rl_amp_cfg import adam_lite_foothold_gru_amp_runner_cfg
    return adam_lite_foothold_gru_amp_runner_cfg()
```

Register:

```text
Instinct-Parkour-Adam-Lite-Foothold-GRU-v0
Instinct-Parkour-Adam-Lite-Foothold-GRU-Play-v0
```

Do not edit existing foothold registrations except for nearby import ordering if required by lint.

## Training Scripts

Optional but recommended new wrappers:

```text
scripts/train_parkour_adam_lite_foothold_gru.sh
scripts/play_parkour_adam_lite_foothold_gru.sh
```

They should mirror the current foothold train/play scripts and only change the task id.

## Verification Checklist

Run narrow checks first:

```bash
python3 -m py_compile \
  src/instinct_mj/rl/module_cfg.py \
  src/instinct_mj/rl/__init__.py \
  src/instinct_mj/tasks/parkour/config/adam_lite/agents/instinct_rl_amp_cfg.py \
  src/instinct_mj/tasks/parkour/config/adam_lite/adam_lite_foothold_gru_parkour_cfg.py \
  src/instinct_mj/tasks/parkour/config/adam_lite/__init__.py

bash -n scripts/train_parkour_adam_lite_foothold_gru.sh
bash -n scripts/play_parkour_adam_lite_foothold_gru.sh
```

If `instinct_rl` is changed, also run:

```bash
python3 -m py_compile \
  /Users/yangrurui/Desktop/instinct_rl/instinct_rl/modules/depth_proprio_gru_encoder.py \
  /Users/yangrurui/Desktop/instinct_rl/instinct_rl/modules/parallel_layer.py \
  /Users/yangrurui/Desktop/instinct_rl/instinct_rl/modules/__init__.py
```

Then verify task registration:

```bash
python -m instinct_mj.scripts.list_envs | rg "Foothold-GRU"
```

If local imports fail due unrelated GUI, matplotlib cache, torch, or `mjviser` issues, do not broaden the patch. Record the exact failure and still run syntax checks.

## Acceptance Criteria

- Existing foothold task ids still resolve to their old config and runner.
- New GRU task ids are registered and use experiment name `adam_lite_foothold_gru`.
- New policy observation group contains baseline actor proprio, one depth frame, and separate GRU proprio history terms.
- Depth input to GRU encoder is one latest/near-latest frame, not an 8 or 10 frame stack.
- GRU receives a real `(B, T, D)` proprio sequence.
- Actor receives GRU latent through the standard encoder output segment.
- PPO update path does not require recurrent hidden-state storage.
- Critic privileged inputs and existing foothold/AMP reward strengths remain unchanged from the foothold baseline unless explicitly justified.

## Later Extensions

After the first version trains and produces stable rollouts, consider:

- Split 48-dim latent into explicit `z_f`, `z_b`, `z_p` heads.
- Add auxiliary decoders:
  - `z_f -> foot height map`
  - `z_b -> base/body height map`
  - `z_p -> next proprioception`
- Add VAE KL loss for `z_p`.
- Add mirror-equivariant channel pairing only after the basic GRU path is working.

Do not combine these extensions with the first implementation unless explicitly requested.
