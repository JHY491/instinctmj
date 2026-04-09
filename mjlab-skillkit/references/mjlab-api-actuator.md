# mjlab.actuator API Reference

## Sources

- `mjlab/docs/source/api/actuator.rst`
- `mjlab/src/mjlab/actuator/__init__.py`

## Public API (grouped by family)

### Basic

- `Actuator`
- `ActuatorCfg`
- `ActuatorCmd`

### Builtin

- `BuiltinActuatorGroup`
- `BuiltinMotorActuatorCfg`
- `BuiltinPositionActuatorCfg`
- `BuiltinVelocityActuatorCfg`
- `BuiltinMuscleActuatorCfg`

### XML

- `XmlMotorActuatorCfg`
- `XmlPositionActuatorCfg`
- `XmlVelocityActuatorCfg`
- `XmlMuscleActuatorCfg`

### Other common variants

- `IdealPdActuatorCfg`
- `DcMotorActuatorCfg`
- `DelayedActuatorCfg`
- `LearnedMlpActuatorCfg`

## How to choose when authoring

- If the MJCF/XML already defines MuJoCo actuators, first check whether the XML / Builtin families can accept them directly.
- If you need classic PD control, use `IdealPdActuatorCfg`.
- If you need a motor model, use `DcMotorActuatorCfg`.
- If you need actuator delay, use `DelayedActuatorCfg`.
- If you need a learned actuator, use `LearnedMlpActuatorCfg`.

## Relationship to action terms

- The action manager does not write inputs directly by joint name. Instead, it routes policy outputs to actuator targets.
- In `mjlab.envs.mdp.actions.*ActionCfg`, `actuator_names` is commonly used to regex-match the target actuators.

## Authoring gotchas

- Action managers route to **actuator** targets, not raw joint names by default; align `actuator_names` regexes with the model’s actuator declarations.
- Do not add a `joint-to-actuator` adapter layer unless the target project already uses that pattern consistently.

## Recommended reading

- Action helpers: `references/mjlab-mdp-builtins.md`
- Entity / scene wiring: `references/mjlab-api-entity.md`
