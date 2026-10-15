# Go2W sim2sim

MuJoCo 和策略控制器是两个 Python 进程，通过真实 `unitree_sdk2_python` 的 `rt/lowstate` / `rt/lowcmd` 通信。仅使用 loopback 和非零 DDS domain，策略 50 Hz、PD 500 Hz；Apple Silicon 可选 MPS 推理。

## 安装与模型

在仓库根目录运行：

```bash
./sim2sim/setup.sh
# 仅在修改 URDF 后重建 MJCF；正常运行直接加载已有 MJCF
.venv-sim2sim/bin/python -m sim2sim.build_model
```

活动模型为 `legged_gym/resources/robots/go2w/urdf/go2w.urdf` 和 `xml/go2w.xml`。四轮使用半径 0.086 m、全宽 0.045 m、rpy=[π/2,0,0] 的 cylinder；总质量 19.523 kg。生成器保留显式惯量并处理固定轮电机的惯量合并。MuJoCo 当前显示碰撞几何。来源与历史问题见 [问题记录](../docs/sim2sim_known_issues.md)。

## 导出和运行

本地 `model_9000.pt` / `model_18000.pt` 来自 TrackinBIT，使用 57/342 输入接口：

```bash
.venv-sim2sim/bin/python -m sim2sim.export_checkpoint \
  --checkpoint /path/to/model_9000.pt --output outputs/model_9000_exported
./sim2sim/run.sh --policy outputs/model_9000_exported/policy.pt \
  --device mps --duration 0 --command 0.5 0 0
```

`--policy` 必填，必须带同目录的 `metadata.json`，或通过 `--metadata` 显式指定；不再自动创建未训练 probe。本仓库训练的 53/318 策略使用 `legged_gym.envs.go2w.export` 导出，运行入口相同。TrackinBIT 默认在观测副本中屏蔽轮速，以匹配其训练输入；导出时可用 `--wheel-observation real` 选择真实轮速。两者的 PD 始终使用真实反馈。

无窗口有限时长运行：

```bash
./sim2sim/run.sh --headless --policy outputs/model_9000_exported/policy.pt \
  --device mps --duration 10 --command 0.5 0 0
```

`--duration 0` 持续运行至关窗或 Ctrl+C。命令为 `[vx, vy, yaw_rate]`；`--schedule` 可传按 start 秒递增的 JSON 列表，如 `[{"start":0,"command":[0,0,0]},{"start":3,"command":[0.5,0,0]}]`。并发运行须指定不同 `--domain`。`--model` 可覆盖 MJCF 路径。

日志默认位于 `logs/sim2sim/<时间>/`，保存策略/metadata 副本及控制器、模拟器、汇总 JSON，记录哈希、实际设备、推理耗时、PD 次数、姿态与 watchdog。不再保存逐步控制/观测 NPZ，也不附带独立评分、回放、基准测试或测试套件。

## 控制约定

腿位置偏移尺度 0.25 rad、Kp/Kd=40/1；轮速度尺度 10 rad/s、Kp/Kd=0/0.5。各关节按名称映射到 SDK；实际动作裁剪和观测布局读取策略 metadata。每次推理生成一个目标，接下来 10 个 2 ms 子步分别用最新 q/dq 计算并限幅力矩。仿真 tick 确认写入 `LowCmd.reserve`，仅用于本地锁步；丢失边界状态会重发，避免改变 50/500 比例。

保留 CRC、有限值/映射/模型维度检查、指令超时、力矩饱和及退出处理。软件锁步和本地运行不代表实机硬实时、训练收敛、粗糙地形或硬件验收。
