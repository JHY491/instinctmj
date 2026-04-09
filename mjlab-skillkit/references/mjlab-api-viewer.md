# mjlab.viewer API Reference

## Sources

- `mjlab/docs/source/api/viewer.rst`
- `mjlab/docs/source/viewers.rst`
- `mjlab/src/mjlab/viewer/__init__.py`

## Public API

- `ViewerConfig`
- `BaseViewer`
- `NativeMujocoViewer`
- `ViserPlayViewer`
- `OffscreenRenderer`
- `EnvProtocol`
- `PolicyProtocol`
- `VerbosityLevel`

## How to use when authoring

- Viewer setup is mainly for play / debugging / scene export, not MDP logic.
- `ViewerConfig` lives on `ManagerBasedRlEnvCfg.viewer`.

## Common viewer choices

- Local desktop debugging: `NativeMujocoViewer`
- Headless or browser interaction: `ViserPlayViewer`
- Screenshots / video / offscreen rendering: `OffscreenRenderer`

## Common `ViewerConfig` fields

- `lookat`
- `distance`
- `elevation`
- `azimuth`
- `origin_type`
- `entity_name`
- `body_name`
- `width`
- `height`

## Authoring suggestions

- Viewer configuration only serves debugging and visualization. Do not mix it into MDP logic.
- Prefer mjlab viewer configs (`NativeMujocoViewer`, `ViserPlayViewer`, `OffscreenRenderer`) over external simulator UI assumptions.
