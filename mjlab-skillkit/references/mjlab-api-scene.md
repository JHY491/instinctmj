# mjlab.scene API Reference

## Sources

- `mjlab/docs/source/api/scene.rst`
- `mjlab/docs/source/scene.rst`
- `mjlab/src/mjlab/scene/__init__.py`

## Public API

- `mjlab.scene.Scene`
- `mjlab.scene.SceneCfg`

## How to use when authoring

- Assemble worlds with `SceneCfg`.
- `SceneCfg` uniformly describes:
  - `terrain`
  - `entities`
  - `sensors`
  - `num_envs`
  - `env_spacing`

## Runtime semantics

- The scene attaches every entity to the same root `MjSpec`.
- Elements inside an entity automatically receive a prefixed namespace:
  - for example, if the entity name is `robot`, then `base_link` becomes `robot/base_link`
- Terrain does not receive an entity prefix and stays in the global namespace.
- `scene.compile()` runs before `Simulation` uploads the result to MuJoCo Warp.
- `scene.initialize()` resolves indices, allocates state buffers, and initializes rendering resources.
- Parallel environments currently share the same `MjModel`:
  - they can run different states in parallel
  - but they **cannot** use different meshes / geoms / kinematic trees per environment
- If the task needs **different** meshes / kinematic trees per environment, mjlab’s current parallel model (shared `MjModel`) cannot represent that; redesign explicitly (for example separate env types or procedural spec edits), do not assume per-env asset variance works out of the box.

## Authoring gotchas

- Avoid `prim_path` / USD scene-graph-only mental models; mjlab composes MuJoCo specs.
- Parallel environments are managed through `num_envs` and `env_origins`, not per-env cloned USD scenes.
- If reset logic previously wrote root state back to an environment origin, it should now read `scene.env_origins`.
- In mjlab, a fixed-base entity participates in multi-environment placement through a mocap wrapper:
  - it is only moved to the correct per-env location after the reset event runs
  - without an appropriate reset term, the fixed-base entity stays at the global origin
- If the task needs cross-entity edits, such as cross-entity tendons / ropes / global sites / shared constraints, prefer `SceneCfg.spec_fn`.

## Scene access patterns

- `env.scene["robot"]`: entity
- `env.scene["terrain_scan"]`: explicitly registered sensor
- `env.scene["robot/trunk_imu"]`: builtin sensor auto-discovered from entity XML

## Easy-to-miss design points

- `SceneCfg.spec_fn` is appropriate for global assembly that a single `EntityCfg` cannot express:
  - cross-entity tendons
  - worldbody-level sites / bodies
  - shared constraints and extra structures across multiple entities
- If logic belongs to scene-wide assembly (cross-entity structure), do not force it into a single `EntityCfg` when `SceneCfg.spec_fn` is the right hook.

## Recommended reading

- Environment shell: `references/mjlab-api-envs.md`
- Sensors: `references/mjlab-api-sensor.md`
- Terrain: `references/mjlab-api-terrains.md`
