# 1. 环境建立与统一数据契约

## 目标与前置条件

建立可复现的 FACTR2 Humble / Python 3.10 环境，冻结后续各 agent 共用的字段、关节顺序、时间语义和路径。无需真机；不安装或改动 W3 的系统 Python。本阶段负责基础环境、契约与 NEXT 配置模板，不实现控制器、adapter 或模型训练。

先读 [执行总览](README.md) 和 [整体规划](../../FACTR2_NEXT_W3_REPRODUCTION_PLAN.md)。

## 代码现状与入口

- FACTR2 根 `README.md`、`.gitignore`。
- `factr2_next/src/factr2_next/setup.py`、`package.xml`、`setup.cfg`。
- `factr2_next/src/factr2_next/factr2_next/config/{record,train,inference,visualize}.yaml`。
- W3 `src/scripts/build_workspace.sh`、`src/README.md`、`src/docs/CONTROL_PIPELINE.md`。
- 上游开发环境是 Jazzy / Python 3.12；本机存在 `/opt/ros/humble` 与 `/usr/bin/python3.10`，目前 FACTR2 没有 `.venv`。可导入和可构建仍需实测。
- `factr2_w3_adapter/` 仅为空目录骨架，第 3 部分才成为可构建的 ROS 包。
- NEXT ROS 节点用 `config_file` ROS 参数读取普通 YAML；trainer 用 `--config`。这些不是同一种参数文件，不能把普通 NEXT YAML 直接当 `--params-file`。

## 冻结的共享契约

将以下内容落成 `docs/interfaces/w3_next_contract.md` 与机器可读 `config/w3/contract.yaml`。契约版本初始为 `w3_next_v1`，后续数据/模型元数据引用它。

| 信号 | 来源 / 输出 | 有效字段 | 单位 |
|---|---|---|---|
| 实测状态 | `/joint_states` | `position` / `velocity` / `effort` | rad / rad/s / Nm |
| 实际控制目标 | `/joint_position_controller/command_state` | `position` / `velocity`，`effort=[]` | rad / rad/s |
| `joint_pos` | `/factr2/{side}/joint_pos` | `position` | rad |
| `joint_vel` | `/factr2/{side}/joint_vel` | `velocity` | rad/s |
| `joint_cmd` | `/factr2/{side}/joint_cmd` | `position` | rad |
| `measured_joint_torque` | `/factr2/{side}/joint_effort` | `effort` | Nm |

以上均为 `sensor_msgs/JointState`。每侧四路输出 `name=[{side}_joint_0,...,{side}_joint_6]`，指定字段长度 7，未使用字段为空。按名字重排，允许输入含另一臂或夹爪，但不可把夹爪当第 8 个模型关节。夹爪有无及负载仍属于数据/模型配置。

adapter 的参数基线：`side` 必须显式为 left/right；`publish_hz=50`、`input_timeout_seconds=0.25`、`max_source_skew_seconds=0.03`、`require_hardware_health=false`（mock）、`hardware_health_timeout_seconds=0.25`。真实采集/部署 profile 必须启用硬件健康门禁。

时间约定：

1. 控制目标时间戳来自 `update(time, period)` 的 `time`，代表本周期写入 command interfaces 的目标，不代表电机真正收到 CAN 帧的时刻。
2. adapter 50 Hz 定时选择最近的有效状态和目标；检查两源接收年龄、ROS 源时间年龄都不超过 timeout，源时间差不超过 30 ms。不在目标插值之外再制造 q_cmd，不把两源差异改 stamp 隐藏掉。
3. 四路输出共用被选中 `/joint_states` 的源 stamp；不使用“当前 timer 时间”给旧状态续命。每个状态源 stamp 最多输出一次。状态和目标数值不变但源 stamp 正常推进，是合法静态样本。
4. 零 stamp（仿真时钟尚未启动）、明显未来 stamp、重复或回退 stamp 不进入有效样本。使用同一 ROS 时钟 / `use_sim_time`；接收超时 watchdog 使用单调时钟。时钟回退时清空缓存并重新等待合法输入。
5. 四流 H5 每行共用 int64 纳秒时间戳；接受的连续片段严格递增。50 Hz / H=50 的训练窗口不能跨 episode 或超过 40 ms 的缺口；缺口切段/拒收，不能删掉中间坏行后拼接成连续窗口。

模型输入 `x=[q,qdot,q_cmd-q]`，单帧 21 维；窗口 `[50,21]`；标签是末帧实测力矩，7 维。归一化仅 fit train，左右分 run。NEXT 三个力矩输出仍遵循上游约定：数值在 `JointState.position` 中（单位 Nm），并补充正确 7 关节 `name`；不要为“更标准”改成 effort 而破坏可视化兼容性。输出 root 固定为 `/next/{side}`。

`/factr2/w3_health` 使用 `diagnostic_msgs/DiagnosticArray`，每侧 status 名为 `factr2/w3/left`、`factr2/w3/right`。OK 的前提见第 3 部分；没有健康消息不是健康。健康 stamp / source stamp / age 字段需在契约中明确。诊断只能判定 bridge 已报告的状态，不能声称知道每个电机真实硬件接收时间。

## 实施步骤与文件范围

1. 新建只影响当前 shell 的环境脚本、依赖约束和使用文档，例如 `scripts/setup_next_env.sh`、`scripts/next_env.sh`、`requirements/next-py310.txt`、`docs/environment.md`。不写 `.bashrc`，不做 `sudo pip`。联网或 apt 需求单独说明；环境脚本可重复执行且不得自动启真机。
2. 首选 `/usr/bin/python3.10 -m venv --system-site-packages .venv`，允许使用 Humble 的 apt Python 绑定；NEXT 科学计算依赖放入 venv。实测确定 NumPy ABI、PyTorch CPU/CUDA wheel、h5py/scipy/colcon/setuptools 版本后固定已验证组合，不盲目照搬 Python 3.12 安装命令。
3. 干净 shell 只 source `/opt/ros/humble/setup.bash` → FACTR2 venv → FACTR2 `install/setup.bash`（构建成功后）。W3 shell 使用 Humble → W3 overlay，禁用 NEXT venv。同一 `ROS_DOMAIN_ID`、兼容 RMW 即可交换标准消息；无需把 W3 overlay 加到 NEXT shell。
4. FACTR2 只构建 `factr2_next`；第 3 部分完成后再加入 adapter。不要构建/启动 Piper、teacher_arm、system_bringup 或安装其硬件 SDK。W3 自定义消息的健康检查器在第 3 部分以**系统 Python + W3 shell**单独运行。
5. 在 `config/w3/left/`、`config/w3/right/` 分别建立 `record.yaml`、`train.yaml`、`inference.yaml`、`visualize.yaml`。Piper 默认配置保留。同步输入 topic/field 与上表完全一致；训练 keys 使用无前缀四键，history=50；独立 train/val 文件；模型默认 stateless LSTM、hidden=128、2 层、head_hidden=256、head_layers=2、dropout=0.1。
6. 为数据、runs、分析输出定义忽略目录和命名约定，如 `data/{side}/{session_id}/`、`runs/next_{side}_{dataset_id}_{timestamp}/`、`reports/generated/`；机器可读 manifest 和模型 metadata 的字段在契约里列出，具体工具在 4/5 实现。

## 验收标准

- [ ] **ENV-01**：干净 NEXT shell 的 `sys.executable` 指向 FACTR2 `.venv`、Python 为 3.10；成功导入 `rclpy,sensor_msgs,diagnostic_msgs,message_filters,ament_index_python,numpy,torch,h5py,scipy,yaml,termcolor`，报告各模块实际路径及版本。无 W3 install 混入，无系统 Python 覆盖。
- [ ] **ENV-02**：只构建 NEXT 核心包成功，四个 console scripts 可发现，`next_train --help` 退出 0。日志明确执行解释器；ROS 节点不以 `--help` 成功作为运行证明，留待 mock 阶段启动。
- [ ] **ENV-03**：两个独立 shell / 进程通过相同 domain 交换一个标准 JointState，字段与 stamp 保真；仅 mock，不启动 bridge/控制器。记录 RMW、domain、QoS。
- [ ] **ENV-04**：左右配置通过脚本检查，四输入 key/topic/field 正确、history=50、单臂、默认 LSTM 和 root 隔离。数据/权重路径可为待填模板，但运行前必须校验，不能静默退回 Piper。
- [ ] **ENV-05**：共享契约有版本、单位、7 关节顺序、输入/输出字段差异、超时/同步/缺口策略；其 YAML 可解析，路径与文档一致。
- [ ] **ENV-06**：依赖锁定、构建/启动步骤可复现；GPU 非必需，CPU 基线必须可用。没有执行任何电机命令。

命令基线（在正确的独立 shell，具体命令写入验收报告）：

```bash
cd /home/dingyj/factr2
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python -m colcon build --base-paths factr2_next/src/factr2_next --symlink-install
source install/setup.bash
ros2 pkg executables factr2_next
ros2 run factr2_next next_train --help
python -c 'import sys, rclpy, torch, h5py; print(sys.executable); print(rclpy.__file__); print(torch.__file__); print(h5py.__file__)'
```

如果 venv/依赖无法安装，交付脚本、配置、契约和准确失败日志；ENV-01/02 不通过。备用容器方案须独立证明标准消息 DDS 互通，不改 W3 Python、不访问 CAN。

## 交付与交接

提交上述配置、脚本、契约、环境文档及 `docs/acceptance/1_environment_and_contract.md`。交给 2/3 的输入是已冻结消息契约；交给 4/5/6 的输入是可运行 NEXT 环境与左右配置，不是训练完成的模型。
