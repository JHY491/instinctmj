# Parkour 深度相机配置说明

本文档说明 Adam SP / G1 Parkour Target AMP 任务中使用的 `NoisyGroupedRayCasterCameraCfg` 深度相机配置，对应代码见 `adam_sp_parkour_target_amp_cfg.py` 与 `g1_parkour_target_amp_cfg.py` 中的 `scene.sensors`。

---

## 1. 概述

- **传感器类型**：`NoisyGroupedRayCasterCameraCfg`（带噪声管道的分组射线投射相机）
- **用途**：在躯干上模拟头戴深度相机，输出「到像平面距离」并经裁剪、模糊、归一化与历史堆叠，供策略或 AMP 使用。
- **输出**：原始 `distance_to_image_plane` 与加噪后的 `distance_to_image_plane_noised` 及其 37 帧历史。

---

## 2. 传感器与挂载


| 参数      | 配置值                                                      | 含义                        |
| ------- | -------------------------------------------------------- | ------------------------- |
| `name`  | `"camera"`                                               | 传感器在环境中的名称，用于观测等引用。       |
| `frame` | `ObjRef(type="body", name="torso_link", entity="robot")` | 挂载在机器人实体的 `torso_link` 上。 |


---

## 3. 成像与光学参数

### 3.1 分辨率与视场（`pattern`）


| 参数       | 值     | 含义        |
| -------- | ----- | --------- |
| `width`  | 64    | 图像宽度（像素）。 |
| `height` | 36    | 图像高度（像素）。 |
| `fovy`   | 58.29 | 垂直视场角（度）。 |


### 3.2 孔径与焦距（内参构造）


| 参数                    | 值                     | 含义                            |
| --------------------- | --------------------- | ----------------------------- |
| `focal_length`        | 1.0                   | 焦距，用于孔径式内参；取 1.0 为归一化约定（见下节）。 |
| `horizontal_aperture` | `2 * tan(89.51° / 2)` | 水平孔径，对应约 89.51° 水平视场。         |
| `vertical_aperture`   | `2 * tan(58.29° / 2)` | 垂直孔径，与 `fovy` 一致。             |


**焦距为何用 1.0**  
内参中像素焦距为：`f_x = width * focal_length / horizontal_aperture`，`f_y` 同理。视场角由 **比值** `focal_length / aperture` 决定，与 `focal_length` 的绝对数值无关。取 `focal_length=1.0` 相当于以「焦距=1」为单位表示孔径，此时孔径即 `2*tan(half_fov)`，公式最简（如 `f_x = width / horizontal_aperture`），且与物理单位解耦。

---

## 4. 位姿与射线


| 参数                    | 含义                                                                                                                |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `ray_alignment="yaw"` | 射线方向只随父体 yaw 旋转，不随 pitch/roll 变化，适合水平朝向的相机。                                                                       |
| `offset`              | 相对 `frame`（torso_link）的位姿：`pos` 平移、`rot` 四元数 (w,x,y,z)，`convention="world"` 表示按世界系约定（前 +X、上 +Z）。当前数值为 G1 头相机标称位姿。 |


---

## 5. 深度输出与裁切


| 参数                        | 值                             | 含义                         |
| ------------------------- | ----------------------------- | -------------------------- |
| `data_types`              | `["distance_to_image_plane"]` | 输出到像平面距离（深度）。              |
| `depth_clipping_behavior` | `"max"`                       | 超过最大距离时裁切到最大值，不产生 inf/nan。 |
| `min_distance`            | 0.1                           | 最近有效测距（米），更近命中被忽略。         |
| `max_distance`            | 2.5                           | 最远有效测距（米），与裁切和归一化范围一致。     |


---

## 6. 噪声与后处理管道（`noise_pipeline`）

管道按配置顺序依次应用，对所有启用的 `data_types` 生效；加噪后类型自动带 `_noised` 后缀。


| 步骤 key                | 配置类                     | 参数                                                                    | 含义                                                               |
| --------------------- | ----------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `crop_and_resize`     | `CropAndResizeCfg`      | `crop_region=(18, 0, 16, 16)`                                         | 从上裁 18、下 0、左 16、右 16 像素；64×36 → 裁后 32×18。未设 `resize_shape` 故不缩放。 |
| `gaussian_blur`       | `GaussianBlurNoiseCfg`  | `kernel_size=3`, `sigma=1`                                            | 高斯模糊，模拟深度噪声与平滑。                                                  |
| `depth_normalization` | `DepthNormalizationCfg` | `depth_range=(0.0, 2.5)`, `normalize=True`, `output_range=(0.0, 1.0)` | 将 [0, 2.5] 米线性映射到 [0, 1]，与 `max_distance` 一致。                    |


---

## 7. 历史与调试


| 参数               | 值                                        | 含义                                                                           |
| ---------------- | ---------------------------------------- | ---------------------------------------------------------------------------- |
| `data_histories` | `{"distance_to_image_plane_noised": 37}` | 对加噪深度做 37 帧历史堆叠，输出在 `sensor.data["distance_to_image_plane_noised_history"]`。 |
| `debug_vis`      | `False`                                  | 不开启该相机的调试可视化（射线/命中等）。                                                        |


---

## 8. 相关代码位置

- 配置类定义：
  - `instinct_mj/sensors/noisy_camera/noisy_grouped_raycaster_camera_cfg.py`
  - `instinct_mj/sensors/grouped_ray_caster/grouped_ray_caster_camera_cfg.py`
  - `instinct_mj/sensors/grouped_ray_caster/grouped_ray_caster_cfg.py`
- 内参与射线计算：`instinct_mj/sensors/grouped_ray_caster/grouped_ray_caster_camera.py`（`_compute_intrinsic_matrices`、`_compute_pinhole_rays_from_intrinsics`）
- 噪声配置与实现：`instinct_mj/utils/noise/noise_cfg.py`、`noise_model.py`

