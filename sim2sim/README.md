# Go2W sim2sim

MuJoCo 和策略控制器是两个 Python 进程，通过真实 `unitree_sdk2_python` 的 `rt/lowstate` / `rt/lowcmd` 通信。仅使用 loopback 和非零 DDS domain，策略 50 Hz、PD 500 Hz；Apple Silicon 可选 MPS 推理。

## 安装与模型

在仓库根目录运行（直接和间接依赖的固定版本统一保存在 `requirements.txt`）：

```bash
./sim2sim/setup.sh
# 仅在修改 URDF 后重建 MJCF；正常运行直接加载已有 MJCF
.venv-sim2sim/bin/python -m sim2sim.build_model
```

活动模型为 `legged_gym/resources/robots/go2w/urdf/go2w.urdf` 和 `xml/go2w.xml`。四轮使用半径 0.086 m、全宽 0.045 m、rpy=[π/2,0,0] 的 cylinder；总质量 19.523 kg。为兼容服务器 Isaac Gym 导入，轮关节使用 `revolute`、位置边界 ±999999 rad，仍按 `kp=0` 的轮速度反馈控制；MJCF 同步该有限边界。生成器保留显式惯量并处理固定轮电机的惯量合并。MuJoCo 默认显示完整 URDF 外观网格和材质；小腿直接复用原始 STL（训练 URDF 使用其等价 DAE 表示），仅将不支持的 DAE 转成共享 OBJ 材质分块。29 个视觉 geom 引用 15 个 mesh asset；视觉网格无碰撞、质量为零，原有 cylinder 碰撞和显式惯量保留。来源与历史问题见 [问题记录](../docs/sim2sim_known_issues.md)。

## 导出和运行

本地 `model_9000.pt` / `model_18000.pt` 来自 TrackinBIT，使用 57/342 输入接口：

```bash
.venv-sim2sim/bin/python -m sim2sim.export_checkpoint \
  --checkpoint /path/to/model_9000.pt --output outputs/model_9000_exported
./sim2sim/run.sh --policy outputs/model_9000_exported/policy.pt \
  --device mps --duration 0 --command 0.5 0 0
```

`--policy` 必填，必须带同目录的 `metadata.json`，或通过 `--metadata` 显式指定；不再自动创建未训练 probe。本仓库训练的 53/318 策略使用 `legged_gym.envs.go2w.export` 导出，运行入口相同。TrackinBIT 默认在观测副本中屏蔽轮速，以匹配其训练输入；导出时可用 `--wheel-observation real` 选择真实轮速。两者的 PD 始终使用真实反馈。

## 按住键盘控制

```bash
./sim2sim/run.sh --keyboard --device mps --duration 0 \
  --policy outputs/go2w_opensource_policy_20261015/exported/policy.pt
```

点击 MuJoCo 窗口，并将鼠标放在中央三维视口内，再按住数字键：`1` 前进、`2` 后退、`3` 左转、`4` 右转；松开或切出窗口即发送零速度命令。支持 `1+3` 等组合，相反方向同时按下会抵消。默认前后速度 1.5 m/s、转向 1 rad/s（集中定义于 `config.py`），可用 `--speed` 和 `--turn-rate` 调整；`Esc` 或关窗退出。

所有有窗口模式均使用 MuJoCo 原生 viewer，保留左右面板。默认开启接触点和接触力箭头；左侧 **Rendering → Contact Point / Contact Force** 可切换，也可在三维视口内按 `C` / `F`。需要数值时，将 **Rendering → Label** 设为 **Force**；显示大小、力箭头缩放可在 **Visualization** 中调整。碰撞体可通过 **Group enable → Geom 3** 显示（外观网格是 Geom 1）。鼠标/相机操作沿用原生 UI；`Tab` / `Shift+Tab` 分别隐藏或显示左/右面板。

键盘控制仅接管三维视口中的 `1`–`4`，在侧边面板输入参数时保留原生键盘行为。GLFW 回调补充原生 Python viewer 缺少的松开事件，无需全局键盘监听权限。不能与 `--headless`、`--schedule` 或非零 `--command` 同时使用。macOS 有窗口运行自动使用 `mjpython`。

松开归零的是策略的速度命令；此前测到的策略自身零命令前漂未在本功能中修正，不能把指令归零等同于物理速度瞬间归零。实现和检查记录见 [问题记录](../docs/sim2sim_known_issues.md#10-按住键盘控制与松开事件)。

## MuJoCo 测试地形

完整地形与键盘控制一起运行：

```bash
./sim2sim/run.sh --keyboard --device mps --duration 0 \
  --policy outputs/go2w_opensource_policy_20261015/exported/policy.pt \
  --terrain mixed --terrain-level 0 --terrain-seed 0
```

`--terrain mixed` 为默认紧凑场地，24×16 m，共 6 块 8×8 m 区域：

| 前进方向（x） | 左侧区域 y=0..8 | 右侧区域 y=8..16 |
|---|---|---|
| 0..8 m | 平地（出生点 x=4,y=4） | 平地 |
| 8..16 m | 起伏路面 | 起伏路面 |
| 16..24 m | 上台阶区 | 下台阶区 |

台阶只保留两个区域；按住 `1` 从平地向前，先经过起伏路面，再抵达台阶。`--terrain-level 0..9` 调整台阶/起伏难度。已移除旧的 200 块课程图入口。

可单独选择 `flat`、`waves`、`slope`、`rough_slope`、`stairs_up`、`stairs_down`、`obstacles`。单种地形为 8×8 m，默认在中央出生：正台阶从平台向外是下阶，负台阶从坑底向外是上阶。`waves` 是本次新增的 sim2sim 平滑二维起伏地形，并非当前训练已启用的类型：纵向波长 2 m、横向波长 3 m，幅值参数 `0.08+0.08*difficulty` m，边缘平滑归零，按 5 mm 量化。幅值经边缘包络和双向叠加缩放，实际极值记录在 `scene.terrain.json`。

参数读取当前 `LeggedRobotCfg.terrain`，Go2W 未覆盖它；相邻 TrackinBIT 仓库的比例、网格间距和生成公式也一致。网格间距 0.1 m、高度单位 0.005 m，台阶高度为 `0.05+0.18*difficulty` 再量化。保留训练中的 `int(0.30/0.1)=2`，实际踏面 0.20 m；没有顺便更改训练代码。紧凑图不复制训练的整图比例；Isaac Gym 的训练地形与课程配置保持不变。

台阶/离散障碍使用静态 box（含低于零面的凹坑），坡面使用 heightfield；不在其下放置会填平凹坑的无限地面。高度样本和参数按训练复现，但 box 硬边并非 Isaac Gym `convert_heightfield_to_trimesh` 的逐三角形副本，坡面最外 0.1 m 延拓边界，接触求解也不同。此功能不宣称两引擎接触动力学严格等价。

地形保留此前的分区配色，地板保留棋盘纹理；按用户最新要求关闭镜面反射。原生 viewer 管理场景绘制，不再维护自制渲染窗口或相机/地形剔除逻辑。

每次运行目录保存 `scene.xml`、`scene.terrain.json`（含 seed、出生点、各块高度哈希和训练配置/生成器哈希）。机器人质量、惯量、控制和观测协议不因地形改变。跌倒高度改用机身到其正下方地形的距离，避免高台阶上的跌倒漏报。实现与验证见[问题记录](../docs/sim2sim_known_issues.md#11-完整外观与训练地形)。

## 无窗口运行

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
