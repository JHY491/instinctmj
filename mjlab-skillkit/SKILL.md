---
name: mjlab-skillkit
description: Use when asked to author new mjlab-native tasks, EnvCfg/scene configs, manager terms, sensors, terrains, RL wiring, mesh import, or task registration. Prefer mjlab public APIs, dict-based managers, nearest task examples, and bundled references; avoid compatibility layers.
---

# mjlab-skillkit (author)

## Scope

- **mjlab Native authoring**: write new mjlab-native tasks, configs, managers, sensors, terrain, RL wiring, and mesh workflows directly.
- Priority: follow mjlab public APIs, existing task patterns, and bundled references in this skillkit.

## Official repositories

- Reference task patterns: `https://github.com/mujocolab/anymal_c_velocity`
- mjlab (read-only reference): `https://github.com/mujocolab/mjlab`

## Documentation priority

- Use local `docs/` in the target workspace first when present (for example `mjlab/docs`).
- Read `references/mjlab-api-pack.md` for the quick API shortlist.
- Read `references/mjlab-api-index.md` to choose the exact mjlab API domain files you need.
- Read `references/mjlab-authoring-workflow.md` for the step-by-step authoring flow.
- For common coding requests, read `references/mjlab-authoring-recipes.md` and follow the matching minimal recipe instead of exploring broadly.
- Then load only the matching `references/mjlab-api-*.md` and `references/mjlab-mdp-builtins.md` files instead of reading the whole API surface at once.
- Do not bulk-ingest the whole `mjlab/docs` tree into context; read only the exact page/signature/example you still need after using the bundled references.
- Fall back to online docs only when local docs are missing or incomplete.

## Reference resolution order

- Prefer raw paths such as `mjlab/docs/source/...` and `mjlab/src/mjlab/...` when a local `mjlab/` checkout exists.
- If there is no local `mjlab/` checkout, use bundled references first:
  - `references/mjlab-api-*.md`
  - `references/mjlab-mdp-builtins.md`
  - `references/mjlab-authoring-workflow.md`
  - `references/mjlab-authoring-recipes.md`
- If the user provides another local mjlab repo path, use that path as the upstream reference root instead of assuming `./mjlab`.
- Use online docs / GitHub only as a last resort for exact upstream signatures or examples missing from both the local checkout and bundled references.
- Do not block authoring just because the current workspace has no `mjlab/` directory.

## Workflow

1. Resolve references in this order: target repo -> local `mjlab/` checkout if present -> bundled skill references -> online docs only if still blocked.
2. Read `references/mjlab-api-pack.md` and lock target APIs.
3. Read `references/mjlab-api-index.md`, then load only the relevant module references:
   - `references/mjlab-api-envs.md`
   - `references/mjlab-api-scene.md`
   - `references/mjlab-api-sim.md`
   - `references/mjlab-api-entity.md`
   - `references/mjlab-api-actuator.md`
   - `references/mjlab-api-sensor.md`
   - `references/mjlab-api-managers.md`
   - `references/mjlab-mdp-builtins.md`
   - `references/mjlab-api-terrains.md`
   - `references/mjlab-api-rl.md`
   - `references/mjlab-api-viewer.md`
   - `references/mjlab-api-tasks.md`
4. If the request involves importing STL / OBJ / other mesh assets, also read `references/mjlab-mesh-import-guidelines.md`.
5. Reuse native mjlab patterns from existing tasks in the target repo first, then from a local `mjlab/` checkout if present, then from bundled references.

## Authoring workflow

1. Read `references/mjlab-authoring-workflow.md`.
2. Read `references/mjlab-authoring-recipes.md`.
3. Identify the artifact type:
   - new task package
   - new `EnvCfg` or scene
   - new manager terms / MDP helpers
   - new sensors / terrain / RL config / registration
4. Choose the closest example before writing:
   - target-repo example first
   - then local `mjlab/` example if available
   - then bundled skill references if no local `mjlab/` checkout exists
   - velocity-style -> `mjlab/src/mjlab/tasks/velocity/`
   - tracking-style -> `mjlab/src/mjlab/tasks/tracking/`
5. Build the config in mjlab-native order:
   - scene / sensors
   - observations
   - actions
   - commands
   - events
   - rewards
   - terminations
   - curriculum / metrics
   - simulation / viewer / episode settings
6. Prefer existing `mjlab.envs.mdp` helpers before adding custom terms.
7. Prefer the smallest edit surface that satisfies the request:
   - add a term before creating a new subsystem
   - modify robot-specific cfg before cloning a base factory
   - reuse an existing command/reward/event helper before inventing a new class
8. If creating a new task, add RL config plus `register_mjlab_task(...)`.
9. Validate syntax and config wiring before completion.

## Authoring request routing

- “Add a reward / penalty” -> `references/mjlab-authoring-recipes.md` + `references/mjlab-mdp-builtins.md`
- “Add an observation / actor-critic input” -> `references/mjlab-authoring-recipes.md` + `references/mjlab-api-managers.md`
- “Add contact / raycast / camera” -> `references/mjlab-authoring-recipes.md` + `references/mjlab-api-sensor.md`
- “Add command / curriculum / reset / DR” -> `references/mjlab-authoring-recipes.md` + `references/mjlab-mdp-builtins.md`
- “Create a new task / new robot env cfg / register a task” -> `references/mjlab-authoring-recipes.md` + `references/mjlab-api-tasks.md` + `references/mjlab-api-rl.md`
- “Import STL / OBJ / mesh assets” -> `references/mjlab-mesh-import-guidelines.md` + `references/mjlab-api-entity.md`
- “Unsure which file to edit” -> first read the “file placement” section in `references/mjlab-authoring-recipes.md`

## Constraints

- Final implementation must be mjlab-native.
- Do not add arbitrary abstractions (extra inheritance/wrappers/major restructuring).
- Do not modify `mujocolab/mjlab` source code.
- Manager configuration must be dict-based (`dict[str, XxxTermCfg]`), not manager `@configclass`.
- Explicitly ban bridge helpers:
  - `manager_terms_to_dict`
  - `AttrDict`
  - `observation_terms_from_class`
- No compatibility layer / adapter shim / transition wrappers.
- Prefer public `mjlab.*` modules and local task examples before inventing new helper layers.
- Prefer task factories such as `make_xxx_env_cfg()` when a reusable base pattern is helpful.
- Reuse `mjlab.envs.mdp` helper functions/configs whenever they already express the needed behavior.
- Keep new code consistent with existing mjlab task structure, config naming, and registration style.

## Execution notes

- Prefer nearest mjlab-native example, not a framework-agnostic rewrite.
- When unsure which mjlab module fits, choose the target module from `references/mjlab-api-index.md` first, then read only the matching reference file.
- For packaging and registration, follow the target project’s conventions (often standalone task package + `mjlab.tasks` entry point + `register_mjlab_task` when that matches existing tasks).
- Managers should be generated via dict factory functions and initialized with `field(default_factory=make_xxx)` when dataclass config style is used.
- If inherited config chains are not supported in target style, flatten them via explicit dict merge/override.
- Follow target project registration conventions (prefer project-level registrar; use `gym.register` only when the target project explicitly requires it).
