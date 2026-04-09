# mjlab.sensor API Reference

## Sources

- `mjlab/docs/source/api/sensor.rst`
- `mjlab/docs/source/sensors/index.rst`
- `mjlab/docs/source/sensors/raycast_sensor.rst`
- `mjlab/docs/source/sensors/rgbd_camera.rst`
- `mjlab/src/mjlab/sensor/__init__.py`

## Public API

### Basic

- `Sensor`
- `SensorCfg`
- `SensorContext`
- `BuiltinSensorCfg`
- `ObjRef`

### Contact

- `ContactSensorCfg`
- `ContactSensor`
- `ContactMatch`
- `ContactData`

### Raycast

- `RayCastSensorCfg`
- `RayCastSensor`
- `RayCastData`
- `GridPatternCfg`
- `PinholeCameraPatternCfg`

### Camera

- `CameraSensorCfg`
- `CameraSensor`
- `CameraSensorData`

## How to use when authoring

- For foot contact or self-collision, prefer `ContactSensorCfg`.
- For terrain height scans, prefer `RayCastSensorCfg`.
- For RGB / depth, prefer `CameraSensorCfg`.
- If MJCF/XML already contains builtin sensors, they can be accessed directly through scene auto-discovery.

## Wiring notes

- Some projects attach contact sensors via `mjlab.utils.spec_config.ContactSensorCfg` on the robot cfg; follow the pattern used in your target repo.
- `sensors=(...)` on the scene side is suitable for explicitly appending sensors.
- Builtin sensors in entity XML are exposed as `entity_name/sensor_name`.
- For contacts that are sensitive to `decimation`, such as self-collision or illegal-contact termination:
  - `history_length` should usually match `decimation`
  - otherwise a short contact may already be gone by the time the policy step reads it
- Sensors such as foot air-time do not always need history, but short-pulse collision detection usually does.

## Additional camera constraints

- `camera_name` non-empty: wraps an existing camera
- `camera_name=None`: creates a new camera
- `parent_body` can attach it to a specific body
- All camera sensors must share the same values for `use_textures`, `use_shadows`, and `enabled_geom_groups`

## Recommended reading

- Scene assembly: `references/mjlab-api-scene.md`
- How raycast / camera observations enter observation groups: `references/mjlab-mdp-builtins.md`
