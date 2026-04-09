# mjlab API Index (load by domain)

This file splits the `mjlab` API into multiple small reference modules that can be loaded on demand.

## Usage order

1. Read `references/mjlab-api-pack.md`.
2. Resolve raw `mjlab/...` paths only if a local `mjlab/` checkout exists; otherwise rely on the bundled reference files below first.
3. After that, load only the matching reference file below for the current problem.
4. Read `references/mjlab-authoring-workflow.md` for end-to-end authoring.
5. If the request matches a common coding action, prefer `references/mjlab-authoring-recipes.md` next.

## Default authoring bundle

For most new manager-based tasks, these five files are a strong default set:

- `references/mjlab-api-envs.md`
- `references/mjlab-api-scene.md`
- `references/mjlab-api-managers.md`
- `references/mjlab-mdp-builtins.md`
- `references/mjlab-api-tasks.md`

## Choose files by problem

| Problem you are solving right now | Read this file | Key API |
| --- | --- | --- |
| Base environment class, EnvCfg fields, episode / decimation | `references/mjlab-api-envs.md` | `mjlab.envs.ManagerBasedRlEnvCfg` |
| Scene assembly, entity prefixes, env origins | `references/mjlab-api-scene.md` | `mjlab.scene.SceneCfg` |
| Simulation timestep, solver, MuJoCo config, DR entry points | `references/mjlab-api-sim.md` | `mjlab.sim.SimulationCfg`, `mjlab.sim.MujocoCfg` |
| Entity access, runtime data fields | `references/mjlab-api-entity.md` | `mjlab.entity.*` |
| Choosing motor / PD / delay / learned actuators | `references/mjlab-api-actuator.md` | `mjlab.actuator.*` |
| Contacts, rays, height scans, cameras | `references/mjlab-api-sensor.md` | `mjlab.sensor.*` |
| Manager dictionaries, `ObservationGroup`, `EventMode`, Null managers | `references/mjlab-api-managers.md` | `mjlab.managers.*` |
| Ready-made action / observation / reward / event / metric / termination helpers | `references/mjlab-mdp-builtins.md` | `mjlab.envs.mdp.*` |
| Flat ground / terrain generators / curriculum terrain | `references/mjlab-api-terrains.md` | `mjlab.terrains.*` |
| PPO runner / vecenv wrapper / train-play flow | `references/mjlab-api-rl.md` | `mjlab.rl.*` |
| Viewer, debug rendering, offscreen export | `references/mjlab-api-viewer.md` | `mjlab.viewer.*` |
| Task registration, task ids, `load_env_cfg` / `load_rl_cfg` | `references/mjlab-api-tasks.md` | `mjlab.tasks.registry.*` |

## Additional pages

- `references/mjlab-authoring-workflow.md`: workflow for writing mjlab-native code directly
- `references/mjlab-authoring-recipes.md`: minimal implementation paths for common mjlab coding requests
- optional if local `mjlab/` exists: `mjlab/docs/source/environment_config.rst`
- optional if local `mjlab/` exists: `mjlab/docs/source/randomization.rst`
- optional if local `mjlab/` exists: `mjlab/docs/source/training/rsl_rl.rst`

## Selection principles

- Express behavior with **official modules** first, then consider task-local custom implementations.
- Express behavior with **manager cfg + mdp helper** combinations first, then consider extra abstractions.
- When unsure, prefer public modules closer to `mjlab/docs/source/api/*.rst`.
