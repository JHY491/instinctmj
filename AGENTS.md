# Agent 指南 - InstinctMJ开发

## 作用范围

- 本指南适用于当前仓库下的全部内容。
- 除非用户明确要求跨项目修改，否则改动应限制在本项目内。
- `InstinctLab` 与 `mjlab` 仓库默认仅用于参考。
- 未经用户明确要求，不得修改 `instinct_rl` 仓库。

## 核心目标

- 保持本项目可作为 `instinct_rl` 的即插即用组件。
- 优先参考 `mjlab` 仓库中相似实现，采用其原生接口与组织方式。
- 当前任务：在 **Adam Lite** 楼梯 parkour 上复现 SSR（https://ssr-humanoid.github.io/）的 imagined foothold guidance（想象落足点引导）。改动只需改 **Adam Lite** 相关代码，除非报错不然不需要改 Adam_SP / G1 相关代码。
- baseline 基准为 commit `301cefc`（纯视觉 `PolicyCfg = InstinctRlEncoderActorCriticCfg` + `WasabiPPO`）。楼梯 baseline 本身可保留 `cref_stair_foothold`；但 **SSR foothold 复现任务不得包含 CReF**（`cref_stair_foothold` / `cref_stepping_stone_foothold` 等分析式落脚 reward）——落脚引导只由 SSR imagined foothold guidance 提供。foothold 任务在去掉 CReF 的前提下，保留 AMP 风格先验等其余 baseline 塑形，并将 SSR `r^f` 作为独立 critic 上的附加 reward；不得用 CReF 与 SSR 双轨并行。

## 迁移规则

- 不得无依据新增“补充函数/辅助函数/包装函数”；原版能直接对应的函数，迁移后应保持同等函数数量与职责划分。
- 不得以“代码更简洁/更优雅”为由压缩原版实现步骤；应按原版逻辑展开，避免隐藏中间过程。
- 在不破坏兼容性的前提下可复用原逻辑代码，重点改写为 mjlab 可接受的表达方式。
- 禁止新增兼容层/桥接层（compat layer、adapter shim、过渡包装）。
- 兜底逻辑必须与源任务一致：源任务有则保留，源任务无则禁止新增。
- reward 必须改为 mjlab reward-term 表达；其他 manager 若存在表达差异，也按 mjlab manager/config 模式迁移。
- 地形及其他核心功能需保持行为一致（reset、command、观测/动作、终止、课程学习）。
- 迁移后必须清理 旧框架 接口残留（import、类型、字段、注释、路径）。
- 清理无效旧路径、过期注释和迁移遗留内容。

## mjlab 对齐要点

- 使用 mjlab 对应导入和命名（如 `ManagerBasedRlEnvCfg`）。
- 将 `@configclass` 配置迁移为字典式 manager 配置对象。
- 以下 manager 均采用字典表达：`rewards`、`observations`、`actions`、`commands`、`terminations`、`events`、`curriculum`。
- 场景使用 `SceneCfg` + MuJoCo 资产配置；不保留 USD/`prim_path` 方案。
- 传感器按 mjlab 方式配置并挂接到机器人实体。
- 统一使用 `entity_name` 字段，不使用旧字段名。

## 兼容性约束

- 将 `InstinctMJ` 视为 `instinct_rl` 的组件层。
- 避免破坏 `instinct_rl` 使用到的训练/回放入口与配置接口。
- 若修复必须改动 `instinct_rl`，先停止并询问用户。

## 代码质量

- 遵循当前项目风格：2 空格缩进、显式类型标注、配置/数据类为中心的结构。
- 保持代码清晰、模块化，避免无关重构。
- 迁移过程中清理无用 import 与死代码。
- 交付前尽量运行可用的 lint/format/tests。
- 简洁工程借鉴 Linus 式偏好（先结构后编码、简单胜过聪明、少无谓抽象）；与「迁移须按原版步骤展开」并存：**简洁指去掉多余复杂度，不是砍掉与源任务对齐的必要步骤**。
- **先想清楚再写**：优先调整数据流、配置边界与调用顺序，使实现路径变短；能用 mjlab 与项目已有模式直接表达的，不绕弯。
- **拒绝「聪明」代码**：禁止以炫技为目的的难读一行式、过深嵌套、隐式副作用；优先直白、可维护的写法。
- **少一层是一层**：除已禁止的兼容层外，不堆无必要的间接层（多余 wrapper、仅转发的配置壳、重复语义的中间类型）；每一层抽象都应对应真实复用或边界。
- **注释写「为什么」**：解释约束、取舍、与任务/物理语义相关的原因；禁止用注释复述代码在做什么。
- **命名可见性**：对外接口命名应让人看出职责；模块内辅助可短，但避免晦涩缩写；类型与函数签名应让人一眼看出输入输出语义。

## Patch 防范原则

1. **整文件替换优先**：不允许输出 diff、不允许“修改以下几行”。
2. **无兼容层**：不允许新增任何 adapter、wrapper、bridge 类。
3. **无条件保留**：不允许写 `if` 来区分新旧框架。
4. **无迭代修补**：一次性输出全部改动，不分多轮。
5. **无注释式清理**：旧代码必须删除，不能注释并标记“待删”。
