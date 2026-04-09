# mjlab-skillkit Rules (author)

This skillkit packages **mjlab-native authoring**: writing new tasks, configs, managers, sensors, terrain, RL wiring, and mesh workflows directly against mjlab.

## Goal

- Author new mjlab-native tasks/components from local or bundled docs and examples.
- Use `mujocolab/anymal_c_velocity` and upstream `mjlab` tasks as structural references when choosing layout and registration style.

## Official references

- Pattern repo: `https://github.com/mujocolab/anymal_c_velocity`
- mjlab repo (read-only reference): `https://github.com/mujocolab/mjlab`

## Mandatory constraints

- Do not modify `mujocolab/mjlab` source code.
- Final implementation must be mjlab-native.
- No compatibility layer / adapter shim / bridge wrappers.

## Manager configuration (hard requirement)

- Do not use manager `@configclass`.
- All managers must be dict-based: `dict[str, XxxTermCfg]`.
- Cover all manager groups: `rewards`, `observations`, `actions`, `commands`, `terminations`, `events`, `curriculum`.
- Do not use bridge helpers:
  - `manager_terms_to_dict`
  - `AttrDict`
  - `observation_terms_from_class`

## Common mjlab naming reminders

- Prefer `entity_name` (not legacy `asset_name`) in configs and docs you write.
- Typical rigid-body data fields: `body_link_pos_w`, `body_link_quat_w`, `body_link_lin_vel_w`, `body_link_ang_vel_w`.

## mjlab API selection cheat sheet

- Environment shell: `mjlab.envs.ManagerBasedRlEnvCfg`, `mjlab.envs.ManagerBasedRlEnv`
- Scene composition: `mjlab.scene.SceneCfg`, `mjlab.scene.Scene`
- Manager configs: `mjlab.managers.*TermCfg`, `mjlab.managers.ObservationGroupCfg`
- Common term helpers: `mjlab.envs.mdp.actions`, `mjlab.envs.mdp.observations`, `mjlab.envs.mdp.rewards`, `mjlab.envs.mdp.events`, `mjlab.envs.mdp.metrics`, `mjlab.envs.mdp.terminations`, `mjlab.envs.mdp.dr`
- Sensors: `mjlab.sensor.ContactSensorCfg`, `RayCastSensorCfg`, `CameraSensorCfg`
- Simulation: `mjlab.sim.SimulationCfg`, `mjlab.sim.MujocoCfg`
- Terrain: `mjlab.terrains.TerrainEntityCfg`, `TerrainGeneratorCfg`
- Training: `mjlab.rl.MjlabOnPolicyRunner`, `RslRlOnPolicyRunnerCfg`, `RslRlVecEnvWrapper`
- Task registration: `mjlab.tasks.registry.register_mjlab_task`

## Scope notes

- Resolve references in this order: target repo -> local `mjlab/` checkout if present -> bundled skill references -> online docs only if still blocked.
- Prefer local/bundled mjlab docs plus nearest existing task example before inventing new abstractions.
- Do not bulk-read the whole upstream docs tree into context; open only exact pages/files needed after the bundled references are exhausted.
- Prefer the smallest matching recipe/edit surface before creating new files or helper layers.
