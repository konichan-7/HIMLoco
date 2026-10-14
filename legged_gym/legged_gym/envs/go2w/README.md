# Go2W HIM 训练

仅注册一个任务 `go2w`，环境为 `Go2w`，配置为 `Go2wRoughCfg / Go2wRoughCfgPPO`。地形、命令、DR 开关与范围，以及 HIM/PPO 参数直接继承仓库原版 `LeggedRobotCfg / LeggedRobotCfgPPO`。

默认使用 trimesh 和地形课程，包含坡面、粗糙坡面、上/下台阶与离散障碍；原版命令课程、heading 模式和横向命令也保留。质量、质心、摩擦、电机强度、Kp/Kd、初始关节、外力、推搡和动作延迟随机化默认开启；link mass / restitution 的开关保持原版默认值。无需切换另一任务来启用 DR。

在约定的 Linux + NVIDIA 训练机、验证过的 Isaac Gym 环境中使用原版入口：

```bash
cd legged_gym/legged_gym/scripts
python train.py --task go2w --headless --rl_device cuda:0 --num_envs 32 --max_iterations 20
```

正式配置继承原版 4096 个环境、100 步 rollout 和训练迭代上限；上述命令仅覆盖小规模 smoke test 的环境数与迭代数。实际环境数和预算需根据 4080S 实测确定。本机不执行 Isaac Gym 训练。

Go2W 专属适配集中在本目录：16 维腿位置/轮速度混合控制、名称映射、53/318/246 接口、轮腿奖励、reset/推搡状态同步、相对标称质心的随机偏移，以及精确选择四个轮接触体。普通四足、基础环境、HIM estimator/actor-critic/PPO/runner 与原版一致。共享训练代码仅增加任务注册，并将原版导出器的 45 维切片改为模型的实际单帧维度。

| 数据 | 布局 |
|---|---|
| policy DOF | FL、FR、RL、RR；每腿 hip/thigh/calf，随后四轮 FL/FR/RL/RR |
| 动作 | 12 个腿位置偏移 + 4 个轮速度目标 |
| actor 单帧 | 命令3 + 角速度3 + 重力3 + 腿位置误差12 + 关节速度16 + 上一动作16 = 53 |
| 历史 | 最新帧在前，6 帧 318 维；reset 清空历史 |
| critic | actor53 + 真值机体线速度3 + 外部扰动力3 + 地形高度187 = 246 |
| HIM target | 使用原版切片：下一步 critic 的 `[3:56]`，去掉命令并追加下一步速度 |
| 控制周期 | 2 ms 物理/反馈控制，10 步 decimation，20 ms 策略 |

在训练机导出完整 source encoder、归一化 latent 与 actor，并生成 sim2sim 所需 metadata：

```bash
python -m legged_gym.envs.go2w.export --checkpoint /path/to/model.pt --output /path/to/exported
```

可添加 `--samples /path/to/parity_samples.npz` 验证真实回放观测；默认只用合成样本检验网络导出一致性。原版 `play.py` 仍可导出 `policy.pt`，Go2W 专属导出入口额外保存 metadata 与 parity 样本。原版 train/runner 的续训及保存行为也保留，本次未继续修改共用框架。

本地张量/网络回归（不调用 Isaac Gym/PhysX）与现有 MuJoCo 接口测试：

```bash
.venv-sim2sim/bin/python legged_gym/legged_gym/envs/go2w/tests/check_go2w.py
.venv-sim2sim/bin/python -m pytest sim2sim/tests -q
```

本地检查不代表 PhysX 地形/接触验证、训练完成或闭环 sim2sim 验收。

2026-10-14 重写后检查结果：12 项 Go2W 张量/网络回归与 5 项 sim2sim 测试通过；全部已跟踪 `rsl_rl/` 文件与 HEAD 逐字节一致，基础环境与 train/play 入口也恢复原版。此记录替代此前双任务方案的当前实现说明；历史审查数据仍保留在 `outputs/`。
