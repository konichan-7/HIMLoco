# Go2W 本机 MuJoCo / Python SDK 测试

本目录搭建两个独立进程，通过真实 `unitree_sdk2_python` 收发 DDS：

```text
MuJoCo bridge -- rt/lowstate --> Python 策略控制器（MPS）
MuJoCo bridge <-- rt/lowcmd  -- Python 策略控制器
```

已在 Apple M4 / macOS arm64 上验证。MuJoCo 物理仿真使用 CPU；TorchScript 神经网络推理使用 PyTorch **MPS/Metal**，默认 `auto` 在此机器选择 MPS，启动日志及报告会记录实际 device。关闭 MPS 算子静默回退。GPU 启用不代表单个小网络必然比 CPU 更快，`doctor` 提供同步后的 CPU/MPS 延迟和数值对照。

**当前没有训练完成的 Go2W HIM 策略。** 默认载入 `interface_probe_untrained`：使用 HIM 对应的网络形状执行推理，腿部保持默认姿态，轮速来自手工滚动参考。它验证模型、SDK、映射、时序和 MPS，不代表 HIM locomotion 或训练策略 sim2sim 已通过。

## 安装

在仓库根目录运行，需要 `uv`、Git、CMake 和本机编译工具：

```bash
./sim2sim/setup.sh
```

脚本创建 `.venv-sim2sim`，在 `.deps` 内编译 Cyclone DDS，不修改系统 Python 或原有训练依赖。主要依赖固定在 `sim2sim/requirements.txt`，安装使用当前实测环境的完整 `requirements.lock.txt`，Python 使用 3.11.16。

固定的上游版本：

- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206)，commit `814556d15970dd2ecf1c9984e845ca02ab07e206`。
- [Cyclone DDS 0.10.5](https://github.com/eclipse-cyclonedds/cyclonedds/tree/2cdd114cbd18340c606573b4cc8dc20cc161ec5a)，Python binding 0.10.2。
- PyTorch 2.8.0、MuJoCo 3.3.7、NumPy 2.2.6。

SDK 文件未修改。SDK factory 的本地网络配置在进程初始化前设置：**loopback（macOS `lo0` / Linux `lo`）、禁用 multicast、仅 localhost peer、默认 DDS domain 71**。本入口不提供实机网卡选项，拒绝 domain 0；不会自动接管机器人运动服务。

## 运行

打开可视化窗口，直到关闭窗口或 Ctrl+C：

```bash
./sim2sim/run.sh --device mps --duration 0
```

启动器自动在 macOS 上使用 `mjpython` 启动 passive viewer，无需手工选择解释器。

有限时长无窗口检查，保存截图：

```bash
./sim2sim/run.sh --headless --device mps --duration 10 --snapshot
```

手工低速前进接口测试：

```bash
./sim2sim/run.sh --device mps --duration 10 --command 0.3 0 0
```

命令顺序为 `vx vy yaw_rate`。本阶段滚动基线要求 `vy=0`，轮速换算只是调试参考。可用负 vx 检查后退，用正/负 yaw_rate 检查差速方向。不同并发测试需使用不同 `--domain`，避免测试参与者相互通信。

加载远端训练并导出的 Go2W 策略：

```bash
./sim2sim/run.sh --device mps --policy /absolute/path/policy.pt \
  --metadata /absolute/path/metadata.json --duration 30 --command 0.3 0 0
```

省略 `--metadata` 时使用策略旁的 `metadata.json`。控制器验证关节顺序和观测契约，不接受原仓库的 270→12 四足策略。metadata 中 `policy_kind="trained_him"` 只是产物标签，不能代替训练/评估证据。

## 当前接口

- 策略顺序：FL、FR、RL、RR，每腿 hip/thigh/calf；随后 FL/FR/RL/RR 四轮。
- 对应 SDK 下标：`[3,4,5,0,1,2,9,10,11,6,7,8,13,12,15,14]`。
- 单帧 53 维：命令 3、角速度 3、重力方向 3、腿位置误差 12、关节速度 16、历史动作 16。
- 历史 6 帧，最新在前；启动时历史为零，依次插入当前帧。轮累计角不进入观测。
- IMU quaternion 为 wxyz。命令、角速度和关节观测尺度见生成的 metadata。
- 物理/低层反馈 500 Hz、策略 50 Hz、SDK 指令发布目标 500 Hz、状态发布 100 Hz。计时使用 monotonic clock；报告记录推理与发包超周期次数。这是本机软件模拟设置，不代表实机固件频率。
- 腿默认姿态 `[0,.8,-1.5]`，位置反馈 `Kp=70,Kd=3`；轮 `Kp=0,Kd=.5`。腿动作尺度 .25 rad，轮动作尺度 10 rad/s，动作裁剪 ±1。这些是本地基线参数，正式训练必须明确采用同一接口或同步修改。
- LowCmd 经 SDK CRC 计算与 bridge 验证；支持位置/速度 stop sentinel。扭矩按本地策略电机限幅裁剪。
- 无有效指令超过 200 ms，bridge 进入仿真阻尼分支；控制器状态超过 250 ms 未更新会报错退出并发送轮零速目标。
- 模拟 LowState 提供关节位置/速度/力矩、IMU 姿态/角速度/加速度和仿真 tick。没有实现足力 ADC、电池、遥控器或固件故障模拟；这些字段保持默认值，LowState CRC 未生成。此入口尚不是完整实机部署状态机。

## 模型生成与资产边界

模型资源统一放在 `legged_gym/resources/robots/go2w/`：`urdf/go2w.urdf` 用于训练模型适配，`xml/go2w.xml` 为 Unitree 官方 MJCF，二者引用同目录的 `meshes/`。官方 MJCF 只调整了 meshdir 相对路径，来源 commit 与哈希记录在 `SOURCE.md`，官方许可在 `LICENSE.unitree_mujoco`。`build_model.py` 读取该资源目录，保留输入文件不变，生成 `.cache/sim2sim/go2w/scene.xml` 和 `model_manifest.json`；`sim2sim/` 不再维护另一份机器人资产。

生成时将官方 `*_wheel_joint` 统一映射为仓库 LowState/策略接口使用的 `*_foot_joint`，重命名 actuator 以支持策略关节顺序查询，并将 MJCF 控制限幅与 `config.py` 对齐。官方模型的几何、惯性、轮碰撞网格和传感器位置作为模型主体保留；本地只覆盖 2 ms timestep、积分器及执行器限幅，添加地面和灯光，并统一加速度计名称。生成的 MJCF 质量约 19.126408 kg，和此前从仓库 URDF 转换的模型不同，后续训练及模型对比应以 manifest 中的模型/资产哈希和质量为准。

这一步只完成了官方 MJCF 到当前 DDS 回放接口的适配和 MuJoCo 编译检查，不代表与 Isaac Gym 训练模型的动力学等价。正式训练前仍需核对轮胎碰撞、惯性、摩擦和执行器参数，并在 Isaac Gym 中检查模型。

## 检查与报告

```bash
.venv-sim2sim/bin/python -m sim2sim.doctor
.venv-sim2sim/bin/python -m pytest sim2sim/tests -q
```

`doctor` 会验证 MPS 运算、SDK CRC，以及非零 HIM 形状网络的 CPU/MPS 输出与延迟，保存 `.cache/sim2sim/environment_report.json`。测试覆盖 SDK/CDR 序列化后 CRC、关节映射、轮速度控制、扭矩饱和、四元数、历史动作时序及模型质量/自由度。

每次运行在 `logs/sim2sim/<时间>/` 保存：

- `controller.json`：实际 device、策略类型/hash、推理分位数、收发数量、超周期次数。
- `simulator.json`：仿真/循环时间、姿态/高度、无效指令、watchdog、饱和次数。
- `summary.json`：两进程结果。
- `inputs/`：该次策略与 metadata 副本、模型 manifest，避免后续重建 probe 覆盖旧运行证据。
- `.npz`：观测历史样本和机体轨迹；`--snapshot` 额外保存 `scene.png`。

`wall_seconds` 和 real-time factor 只衡量物理循环，`total_wall_seconds` 另含截图与退出处理。正式策略验收还需独立评估跟踪误差、跌倒率、命令切换和停止行为。
