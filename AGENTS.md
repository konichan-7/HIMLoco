# 项目背景与协作约定

更新日期：2026-10-13。本文件记录用户已确认的目标和硬件条件，供后续在本仓库工作的 agent 使用。具体实施路线见 [技术路线文档](docs/go2w_locomotion_sim2sim_roadmap.md)。

## 1. 最终目标

在 **Unitree Go2W 轮腿机器人上复现 HIMLoco 方法**，依次完成训练、MuJoCo sim2sim 和实机部署。最终产物需要包括机器人适配、可复现的训练配置、完整推理模型、仿真评估以及可在机器人上运行的部署代码。

保持 HIM 的方法主体：历史本体观测、显式速度估计、隐式响应 latent、source/target encoder、prototype/Sinkhorn 对比学习，以及 HIO 与 PPO 的优化关系。Go2W 的动作、观测、奖励和电机模型适配需单独记录，不将参考项目的 DreamWaQ/VAE 算法替换为本项目的训练主体。

原论文主要验证普通四足平台；本项目结果应准确表述为 HIM 方法的 Go2W 迁移复现，论文原始指标与本项目实测结果分别报告。

## 2. 用户已确认的机器分工

| 环境 | 已确认条件 | 工作内容 |
|---|---|---|
| 当前开发机 | 本仓库所在本机 | 代码开发、模型/接口检查、MuJoCo 调试与 sim2sim |
| 远端训练机 | Linux + NVIDIA 4080S | Isaac Gym 环境验证、HIM/PPO 训练、评估与导出；模型训练不在本机进行 |
| Go2W 机载计算机 | **NVIDIA Orin NX** | 将 Python 部署代码、模型与配置拷贝到该设备，由 Orin NX 调用 Python 版 Unitree SDK 控制机器人 |

Orin NX 的系统、JetPack/CUDA、推理运行时、网络接口和机器人固件版本尚未现场确认。远端训练机的发行版、驱动、Isaac Gym 安装状态及连接信息也尚未确认。不要推测 SSH 地址、账号、设备路径或软件版本。

训练环境和机载推理环境独立管理。用户已指定部署使用 **Python 版 Unitree SDK（`unitree_sdk2_python` / `unitree_sdk2py`）**。部署程序不依赖 Isaac Gym；根据 Orin NX 实测兼容性和延迟选择 Python 推理运行时，并验证与训练模型的输出一致性。

## 3. 当前仓库起点

以下为 2026-10-13 的检查快照，后续修改前重新核对实际代码与 Git 状态：

- 基准提交：`ef289acaa62795009363b7b819c9186690630441`。
- 已注册 `a1`、`go1`、`aliengo`；还没有 Go2W 训练任务。
- 训练使用当前仓库自带的 `legged_gym` 和 `rsl_rl`，不要直接换成上游最新版。
- 原始接口为 12 维动作、45 维单帧观测、6 帧历史；导出器含写死的 45 维切片。
- Go2W 资源在 `legged_gym/resources/robots/go2w/`，当前为未跟踪文件；远端交付时需要包含这些资源。
- Go2W 已统一为 `urdf/go2w.urdf`、`meshes/`、`xml/go2w.xml` 的资源布局，URDF 使用相对网格路径，不依赖 ROS package 解析。官方 MJCF 已从 `sim2sim/assets/` 迁入此目录；来源与哈希见 `SOURCE.md`，许可见 `LICENSE.unitree_mujoco`，sim2sim 生成器从该统一目录读取。
- Go2W 有 12 个腿关节和 4 个连续旋转轮关节；轮关节名是 `FL/FR/RL/RR_foot_joint`。
- 本机已搭建 `sim2sim/`：从资源目录中的官方 MJCF 派生本地 MuJoCo 场景，两个独立 Python SDK 进程在 loopback DDS 通信，TorchScript 支持 Apple Silicon MPS。运行说明见 [sim2sim/README.md](sim2sim/README.md)。URDF 与官方 MJCF 的质量/碰撞并非已对齐，资源整理不代表动力学一致性验收。
- 本地验证使用标记为未训练的接口 probe；尚未实现 Go2W HIM 训练或 Orin NX 实机部署，也没有完成训练策略 sim2sim 与实机验收。

主要代码入口：

| 路径 | 职责 |
|---|---|
| `legged_gym/legged_gym/scripts/train.py` | 训练入口 |
| `legged_gym/legged_gym/envs/__init__.py` | 任务注册 |
| `legged_gym/legged_gym/envs/base/legged_robot.py` | 观测、控制、奖励、reset、随机化与课程 |
| `rsl_rl/rsl_rl/modules/him_actor_critic.py` | HIM、actor 与 critic |
| `rsl_rl/rsl_rl/modules/him_estimator.py` | 速度回归与对比学习 |
| `rsl_rl/rsl_rl/algorithms/him_ppo.py` | HIM/PPO 更新 |
| `rsl_rl/rsl_rl/runners/him_on_policy_runner.py` | rollout、日志与 checkpoint |
| `legged_gym/legged_gym/utils/helpers.py` | 参数解析与策略导出 |

## 4. Go2W 适配与三端统一接口

- 建议使用 16 维策略动作：腿为位置偏移，轮为速度目标；两组动作采用各自尺度、增益与限幅。
- 轮关节不设置回零位置弹簧；累计轮角不直接进入姿态误差/位置限位惩罚。位置观测可屏蔽轮角或采用有明确理由的周期表示，训练与部署保持一致。
- 当前路线建议单帧 53 维、6 帧共 318 维；这是待实施的接口方案，不是当前代码已支持的规格。正式训练前冻结布局与 metadata。
- 观测重排必须同步 HIM target 的字段选择：原版 `next_critic_obs[:, 3:num_one_step_obs+3]` 依赖命令位于前三维，实际构造“下一步去掉命令的本体观测 + 下一步真值机体线速度”。不能仅改维度或观测顺序而保留旧切片；速度监督、target 输入和 actor 输入分别显式定义。
- 按关节名称建立 Isaac Gym、MuJoCo、策略和 SDK 的映射；不能假定它们使用相同顺序。
- 统一观测尺度、机体 frame、四元数顺序、历史排列/reset、历史动作时序、控制周期、延迟、默认姿态和力矩饱和。
- 部署 actor 使用 IMU、编码器、历史动作和用户命令；仿真真值和地形信息仅用于训练监督/critic 或独立评估。
- 用户已确认初版训练沿用原版 critic 的特权信息组成，不新增足端/轮端接触力输入：本体观测 + 真值机体线速度 + 外部扰动力 + 地形高度。按候选 53 维单帧及 187 个高度点，critic 为 `53+3+3+187=246` 维；“无接触力输入”不意味着删除原有外部扰动力，或停止在 reward/终止判定中读取接触力。
- 导出完整 HIM source encoder + latent 归一化 + actor，保留对照输入/输出样本；critic、target encoder、prototype 不进入部署推理。

## 5. 官方 SDK 与 Orin NX 部署

官方依据：

- [Go2-W SDK 开发指南](https://support.unitree.com/home/zh/Go2-W_developer/Operation_guide)
- [Go2-W 底层服务接口](https://support.unitree.com/home/zh/Go2-W_developer/Basic_services)
- [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [Unitree SDK2 Go2W 例程](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2w/go2w_stand_example.cpp)
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)

部署代码、模型和配置需要拷贝到 Orin NX，由 Python 机载部署进程通过 `unitree_sdk2_python` 订阅 `rt/lowstate` 并发布 `rt/lowcmd`，使用 Go2W 对应的 `unitree_go` 消息。SDK 语言绑定已经确定为 Python；实际 Python 版本、SDK commit、依赖和推理运行时写入部署记录。

官方电机编号为：

| 部位 | SDK 下标 |
|---|---|
| FR hip/thigh/calf | 0/1/2 |
| FL hip/thigh/calf | 3/4/5 |
| RR hip/thigh/calf | 6/7/8 |
| RL hip/thigh/calf | 9/10/11 |
| FR/FL/RR/RL wheel | 12/13/14/15 |

IMU quaternion 为 `wxyz`；Isaac Gym 的根姿态为 `xyzw`。实机需核对轮速符号、零位和 IMU frame。

首版目标控制语义：腿发送位置目标与 `kp/kd`；轮发送速度目标、`kp=0`、`kd=Kv`。仿真模拟对应反馈力矩与饱和；避免在启用电机反馈增益的同时重复发送同一 PD 力矩。高层 `SportClient.Move` 不作为 HIM 关节动作的执行接口。

SDK 发包频率、策略推理频率与固件内部伺服频率分别处理。官方例程的 500 Hz 发包及路线候选 50 Hz 策略需要实测后冻结，不视为已经验证的机载性能。

本机 macOS 已验证 Python SDK 的原生 DDS 仿真通路；当前入口限制 loopback 和非零 domain。后续将共用策略核心在 Orin NX/Linux 验证，补全实机控制权切换和状态机，再做实机测试。仿真与实机使用隔离的 DDS domain/interface 配置。

## 6. Reward 与轮腿适配参考

### 6.1 DreamWaQ_Go2W：reward 与轮腿控制参考

用户指定参考：[ShengqianChen/DreamWaQ_Go2W](https://github.com/ShengqianChen/DreamWaQ_Go2W)。重点借鉴其 **Go2W reward 和轮腿控制适配，尤其是相对普通 Go2 四足的变化**。

本轮源码检查固定到 commit `b64f7b90856fc71595870cb283ce29dd0f4d6f0f`：

- [Go2W 配置与 reward scales](https://github.com/ShengqianChen/DreamWaQ_Go2W/blob/b64f7b90856fc71595870cb283ce29dd0f4d6f0f/legged_gym/legged_gym/envs/go2w/go2w_config.py)
- [Go2W 控制、观测与 reward 实现](https://github.com/ShengqianChen/DreamWaQ_Go2W/blob/b64f7b90856fc71595870cb283ce29dd0f4d6f0f/legged_gym/legged_gym/envs/go2w/go2w_robot.py)
- [继承的基础 reward 配置](https://github.com/ShengqianChen/DreamWaQ_Go2W/blob/b64f7b90856fc71595870cb283ce29dd0f4d6f0f/legged_gym/legged_gym/envs/base/legged_robot_config.py)

已核实的参考设计与本项目移植关注点：

| 参考设计 | Go2W 相对普通四足的关注点 |
|---|---|
| `dof_acc` 排除轮关节，单设 `wheel_acc` | 腿与轮的速度/加速度范围不同，应分组惩罚与调权重 |
| `dof_vel` 意图排除轮关节 | 正常行驶需要轮子持续旋转，不应被普通关节速度惩罚压制 |
| `stand_still`、`dof_error` 排除轮角 | 累计轮角不代表支撑姿态误差；零命令时另评估轮速、机身漂移和 yaw |
| `hip_pos` 约束髋部姿态 | 关注侧向稳定与姿态保持，同时避免过强约束妨碍踏步 |
| 基座高度、姿态、碰撞、绊轮、动作变化惩罚 | 结合轮半径、腿姿态与轮腿接触重新定义和调权重 |
| 此版本未启用腾空时间/抬脚奖励 | 平地滚动先验证持续支撑；后续跨障碍任务再定义摆动/抬轮奖励 |

参考配置的高度目标为 0.4 m，腿位置动作尺度为 0.25、轮速度尺度为 10；这些仅是参考值，需要与本仓库模型和实机参数核对后使用。其 `only_positive_rewards=True` 也需与当前 HIM 配置及各奖励量级一起评估。

借鉴时同时阅读实际函数、有效继承配置、单位、控制周期和关节索引，不机械复制权重。源码没有给出独立的滑移奖励实现；若本项目增加滑移项，标记为新增设计，并针对接地点相对地面运动计算，避免把正常轮心运动当作打滑。

已发现参考实现中的原地状态修改：`_reward_dof_vel()` 将 `self.dof_vel` 的轮速度置零，且该张量是仿真状态的 view；观测处理中也存在直接清零 `self.dof_pos` 的代码。移植时使用掩码/独立观测张量，reward 不修改物理状态，以免影响后续 wheel acceleration、观测和控制。

参考项目的观测维度、history、DreamWaQ 网络和训练超参数不能直接替代本仓库 HIM 接口。借用实现时保留来源归属并核对适用许可。

### 6.2 HIMLoco-for-Go2W：HIM 轮腿迁移参考

用户新增指定参考：[TrackinBIT/HIMLoco-for-Go2W](https://github.com/TrackinBIT/HIMLoco-for-Go2W)。2026-10-13 的源码对比固定到 commit `011693738c61603c3f22f2bce755098dd36fa7eb`，对照官方 HIMLoco commit `ef289acaa62795009363b7b819c9186690630441`。完整证据见 [源码对比报告](outputs/himloco_go2w_comparison_20261013/comparison.md)，同目录保留文件清单、哈希和目录对齐后的 diff。

主要源码：

- [Go2W 配置](https://github.com/TrackinBIT/HIMLoco-for-Go2W/blob/011693738c61603c3f22f2bce755098dd36fa7eb/legged_gym/envs/go2w/go2w_config.py)
- [基础环境中的观测、控制与 reward](https://github.com/TrackinBIT/HIMLoco-for-Go2W/blob/011693738c61603c3f22f2bce755098dd36fa7eb/legged_gym/envs/base/legged_robot.py)
- [HIM estimator 与 target 切片](https://github.com/TrackinBIT/HIMLoco-for-Go2W/blob/011693738c61603c3f22f2bce755098dd36fa7eb/rsl_rl/rsl_rl/modules/him_estimator.py)
- [完整 HIM 导出器](https://github.com/TrackinBIT/HIMLoco-for-Go2W/blob/011693738c61603c3f22f2bce755098dd36fa7eb/legged_gym/utils/helpers.py)
- [MuJoCo 回放](https://github.com/TrackinBIT/HIMLoco-for-Go2W/blob/011693738c61603c3f22f2bce755098dd36fa7eb/mujoco/pdandrl.py)

已核实的设计与适用边界：

| 项目 | 参考实现与本项目约定 |
|---|---|
| HIM/PPO 主体 | 整个 `rsl_rl/` 的 25 个文件与上述官方原版逐字节一致；网络输入/输出通过配置扩大。算法文件一致不代表训练输入语义一致 |
| 动作与控制 | 16 维，腿位置偏移尺度 0.25、`kp=40/kd=1`；轮速度尺度 10、`kp=0/kd=0.5`。借鉴混合控制语义，具体参数仍需模型/实机核对 |
| actor/critic 接口 | 单帧 57 维、6 帧 342 维；critic 262 维，额外四轮 xyz 接触力仅供 critic。不能直接替代本项目候选 53/318 接口 |
| 轮角位置槽 | 16 维位置块中四个轮槽被置零，保留的是布局槽位，没有使用真实轮角；之后统一加位置噪声。保留完整 DOF 布局可能便于复用代码，但这是工程解释，未找到作者动机或性能消融证据 |
| reward | 借鉴 `hip_default`、排除轮角的 `stand_still/run_still`、滚行姿态约束；此版本没有独立 `wheel_acc` 或滑移 reward，不与 DreamWaQ 参考混淆 |
| 组织方式 | Go2W 子类基本为空，适配直接修改基础环境，并移除普通四足任务。本项目应隔离 Go2W 适配并保留原任务 |
| 模型与部署 | 附 checkpoint、日志和完整 source encoder + 速度/归一化 latent + actor 的 TorchScript；本地已验证对应参数/抽样输出一致及 MJCF 可加载，不代表闭环 sim2sim 验收。没有 Orin NX Python SDK 实机部署入口 |

**本项目采用决定**：初版不移植参考实现的 12 维足端/轮端接触力 critic 输入，先建立无该输入的 HIM Go2W 基线。“有／无接触力”仅列为后续待做消融，不作为初版必做训练，也不表示已实现或运行。

轮角选择原则：对近似轴对称轮的普通滚行，累计角度及任意编码器零点不直接表征支撑姿态，不应原样当腿关节角使用。首版优先使用 12 个腿位置误差、16 个关节速度和 16 个历史动作，维持 53/318 候选方案；若保留四个轮位置占位槽，观测和噪声均明确掩码，并保持训练/部署一致。只有在需要轮转动相位或短时相对里程信息时，才评估周期表示或角增量并单独验证；不因参考模型保留槽位就推断轮角必需。

必须处理的参考实现问题：

- **共享状态污染**：观测原地清零 `self.dof_pos` 轮分量，`_reward_dof_vel()` 原地清零 `self.dof_vel` 轮分量；二者是 Gym state tensor 的 view。reward 先于观测，导致训练轮速观测与 `last_dof_vel` 被清零，而 MuJoCo 输入真实轮速；`dof_acc` 未排除轮，计算也被污染。移植时用独立张量/索引，不修改共享状态缓冲，不声称未经实测的 PhysX 内部状态重置。
- **target 语义变化**：参考 actor 顺序是 `[角速度3, 重力3, 命令3, 位置16, 速度16, 动作16]`，但 estimator 仍取 `next_critic_obs[:, 3:60]`；由原版排除命令变成排除角速度、保留命令。速度监督 `57:60` 本身仍正确。没有证据说明这是有意的算法改进；保留命令可能让 latent 更容易按任务命令聚类，移除角速度则减少直接动态响应信息，这些影响是待实验验证的假设。
- **target 修正原则**：若维持“角速度、重力、命令”顺序，令 `D=num_one_step_obs`，target 显式拼接 `next_critic_obs[:, :6]`、`next_critic_obs[:, 9:D]`、`next_critic_obs[:, D:D+3]`，即排除命令并保留角速度/重力/关节/动作/下一步速度；需与终止 transition、reset 和观测尺度一起验证。不得仅把固定切片中的 45 换为 57 或 53。
- **sim2sim 差异**：训练每 5 ms 重算反馈力矩，MuJoCo 每 20 ms 算一次并保持 4 子步；髋/大腿限幅训练 23.7 Nm、MJCF 35.55 Nm；航向误差到 yaw 速度的增益/限幅也不同。回放 YAML 的 `timestep=0.002` 未使用，实际 MJCF 为 0.005 s，策略仍 50 Hz。
- **导出限制**：写死 57 维并只支持单样本，导出文件为 `policy.pt`，演示配置指向 `policy_best.pt`。正式版本需动态维度、metadata 和一致性样本。更改轮速输入、target 定义或 53/318 布局后，不能把所附 57/342 策略自动认定为适配新接口。

## 7. 实施与验证约定

- 顺序：模型与低层控制检查 → 统一接口 → 远端小规模训练验证 → 正式训练 → 导出一致性 → MuJoCo/DDS 验收 → Orin NX 回放与性能验证 → 实机逐步验收。
- 记录代码/SDK commit、策略及资产哈希、配置、周期、软件环境、命令和地形测试范围。
- 区分“代码完成”“训练运行/完成”“sim2sim 通过”“代码已拷贝到 Orin NX”“机载程序运行”“实机控制通过”，只报告有证据的阶段。
- 不以 reward 上升、checkpoint 存在、部署编译成功或仿真视频替代完整验收。
- 修改接口时同步环境、网络维度、噪声切片、HIM target/速度监督时序、导出器、metadata、MuJoCo 和 SDK 解码，并做有意义的一致性与回归检查。
- 保留现有普通四足任务和用户已有资源；修复已确认的问题时记录原因，避免顺带大规模改动训练框架。

待做消融：**critic 有／无足端接触力特权输入**。无接触力组使用初版 critic；有接触力组仅在 critic 末尾增加四个轮接触体的 xyz 合力（12 维），按候选接口为 246／258 维。两组均保留原有外部扰动力，actor、HIM source/target、reward、控制、随机化、课程和训练预算保持一致，仅调整接触力输入及必需的 critic 输入层；固定代码/资产版本，使用配对的多个训练种子和同一独立评估集。比较成功率、速度误差、停止漂移、地形/扰动通过率及训练稳定性，不仅比较 reward 或 value loss。接触体名称、坐标系、尺度和裁剪在消融实施前冻结；详细计划见 [技术路线的消融条目](docs/go2w_locomotion_sim2sim_roadmap.md#51-待做消融critic-有无接触力特权输入)。
