# W3 双臂机械臂 FACTR2 NEXT 复现规划

## 目的与边界

本文依据本地 `factr2_next` 源码、论文《FACTR 2: Learning Force Sensing and Force-Aware Policies for Any Robot Arm》（arXiv:2606.12406），以及现有 `w3_dual_arm_ws` 控制代码，规划在没有末端力/力矩传感器的 W3 真机上复现 NEXT 外力估计。

当前真机的基础标定、CAN bridge 和控制器启动已可直接使用。本项目的新增工作集中在：

1. 让 W3 关节位置控制器公开其内部插值后的目标状态。
2. 将 W3 的关节观测和控制目标转换为 NEXT 约定的 ROS 话题。
3. 在无接触自由运动数据上训练模型，并进行只读在线推理。

第一阶段只输出外力估计和诊断量，不将估计力矩送回控制器，不实现双向力反馈或基于接触状态的运动策略。NEXT 的关节外力估计不是六维末端 wrench 测量，不能直接等同于笛卡尔空间力。

## 1. 整体技术路线

NEXT 先学习机械臂在无外部接触时，为完成自身运动所表现出的电机/关节力矩。模型输入当前运动状态与控制目标，输出自由空间力矩预测；真机测得力矩减去预测值后，剩余项作为外部关节力矩估计：

```text
x_t = [q_t, qdot_t, q_cmd_t - q_t]
tau_free_hat(t) = f_theta(x_(t-H+1):t)
tau_ext_hat(t) = tau_measured(t) - tau_free_hat(t)
```

上游默认 `H=50`，目标记录频率为 50 Hz，即窗口覆盖约 1 秒。代码默认使用 2 层、单向、stateless LSTM，窗口末帧的目标是该时刻实测力矩，不是未来力矩。模型可配置为 MLP 或 GRU，但第一版应保持默认 LSTM，减少相对上游的变量。

完整数据路径：

```text
DM 电机反馈
  -> W3 bridge /w3_robot_bridge_node/state
  -> DMHardwareInterface 标定转换
  -> /joint_states (q, qdot, tau_measured)
  -> W3 实时目标接口 (q_cmd, qdot_cmd)
  -> factr2_w3_adapter 统一名称、顺序、时间戳和字段
  -> next_record 写 H5
  -> next_train 训练 tau_free_hat
  -> next_infer 在线计算 tau_measured - tau_free_hat
  -> 外力估计、接触指示和可视化
```

## 2. 工作区和环境隔离

### W3 工作区

位置：`/home/dingyj/w3_dual_arm_ws`

- 保持现有 Ubuntu 22.04、ROS 2 Humble、系统 Python 3.10、colcon/ament 环境。
- 继续负责 CAN FD bridge、ros2_control、标定后关节状态和运动命令。
- 只需添加目标状态只读发布接口和相关测试，不安装 PyTorch/H5Py，不保存模型或训练数据。
- 真机仍按当前工作区文档启动，不让 NEXT 启动或停止 bridge/控制器。

### FACTR2 工作区

位置：`/home/dingyj/factr2`

- 上游源码留在 `factr2_next/`，在其旁边增加独立 `factr2_w3_adapter/` 包。
- NEXT 的虚拟环境、build/install/log、H5 数据、训练 run、checkpoint 和分析结果均留在此目录。
- W3 和 NEXT 使用 ROS 2 话题通信，不将 PyTorch 依赖写入 W3 包，也不把控制包并入 NEXT Python 环境。

### Python/ROS 版本策略

W3 当前已验证的组合为 ROS 2 Humble + Python 3.10。上游 `factr2_next/README.md` 的开发组合为 ROS 2 Jazzy + Python 3.12，并以 `uv` 建环境；不能直接假定 Python 3.12 虚拟环境可导入 Humble 的 `rclpy`。

首选先在 Python 3.10 下为 FACTR2 建立独立 venv，并通过 ROS Humble 系统环境提供 `rclpy` 等绑定；只在 venv 中安装模型/数据科学依赖，例如与当前 NumPy ABI 兼容的 `torch`、`h5py`、`scipy`、`pyyaml`、`termcolor`。启动 NEXT 终端时按顺序 source Humble、W3 overlay、再激活 venv，并检查 `sys.executable`、`rclpy`、`torch`、`h5py` 的实际导入路径。

若核心依赖无法在 Python 3.10 环境稳定安装，不要改动 W3 系统 Python。备用方案是把 NEXT 运行在单独的 Humble/Jazzy 容器或独立 ROS 2 Jazzy 环境中，通过 DDS 与 W3 通信，并验证跨发行版消息互通；容器方案需要处理主机网络、ROS_DOMAIN_ID、DDS 实现及 CAN 权限，第一版不让容器直接访问 CAN。

建议分别构建：W3 在自己的工作区按现有 Humble 流程构建；FACTR2 在 `factr2/` 的 venv 中构建 Python 包。不得同时 source 彼此的 install overlay，以免 Python 包、ament index 或 CMake 路径交叉污染。两个独立终端通过同一 ROS domain 通信。

## 3. 真机已有接口与新增接口

### 已有 W3 状态

W3 bridge 发布：

```text
/w3_robot_bridge_node/state  w3_robot_bridge/msg/MotorStateArray
```

每个 motor state 包含 `channel`、`motor_index`、`position`、`velocity`、`torque`、`online`、`enabled` 和 `error_flags`。左臂为 channel 0/can0，右臂为 channel 1/can1，关节 slot 是 0..6；slot 是零起始，不等于 CAN ID。该话题适合查看原始电机反馈、在线状态和故障码，不建议 NEXT 直接使用它：标定、关节方向、URDF 坐标窗口等由 ros2_control 硬件接口负责转换。

控制器侧的 `/joint_states` 为 `sensor_msgs/JointState`，包含经标定后的关节 `position`、`velocity` 和 `effort`。`effort` 是电机反馈力矩按关节轴方向换算后的值。NEXT 第一版使用它作为 `measured_joint_torque`，同时保留原始 W3 state 用于审计。

### 需要新增的控制目标状态

已有 `/joint_position_command` 是轨迹输入。`JointPositionController` 在 300 Hz 控制循环中按段线性插值，目标位置和速度会随时间变化，因此不能把输入轨迹的稀疏点直接当作 `q_cmd`。

规划新增：

```text
/joint_position_controller/command_state  sensor_msgs/JointState
```

字段约定：`name` 为控制器当前关节顺序；`position` 是本周期最终采用的 `pos_des`；`velocity` 是插值斜率 `vel_des`；`header.stamp` 使用控制周期时间；`effort` 留空。它只读发布，不增加订阅命令或改变控制器行为。由于它从 ros2_control update loop 发布，应使用 `realtime_tools::RealtimePublisher`，不能在实时循环内做普通阻塞发布。

位置控制器未 active 时不应发布伪造的活动目标。NEXT 真正采集时需要位置控制器 active 并有有效的 `command_state`。gravity 模式可以用于检查静态反馈和力矩信号，但不能提供与 NEXT 输入定义一致的控制目标历史。

## 4. W3 到 NEXT 转换接口

在 `factr2/` 新建独立 Python 包 `factr2_w3_adapter`，订阅 `/joint_states` 和 `/joint_position_controller/command_state`，不订阅控制命令，不发布任何 W3 命令。

### 输入映射

| NEXT 信号 | W3 来源 | `JointState` 字段 |
|---|---|---|
| `joint_pos` | `/joint_states` | `position` |
| `joint_vel` | `/joint_states` | `velocity` |
| `joint_cmd` | `/joint_position_controller/command_state` | `position` |
| `measured_joint_torque` | `/joint_states` | `effort` |

### 输出映射

以左臂为例：

```text
/factr2/left/joint_pos
/factr2/left/joint_vel
/factr2/left/joint_cmd
/factr2/left/joint_effort
```

右臂使用 `/factr2/right/...`。四路都发布 `sensor_msgs/JointState`，数值放在上表指定字段中。每个消息的 `name` 固定为 `left_joint_0 ... left_joint_6` 或右臂对应名称，并按 0..6 排列。适配器按 `name` 重排输入，不依赖 W3 publisher 的数组顺序。四路采用相同时间戳发布，50 Hz 定时采样最新同步状态。

适配器应要求两路输入都已收到且未超过超时阈值（初始建议 0.25 秒），并检查 7 个关节齐全、向量长度匹配、数值有限。任意输入缺失、过期或包含 NaN/Inf 时，本周期不发布有效 NEXT 样本并输出限频告警。记录实际输出频率和无效帧计数。适配器可用 ROS 参数选择 `side`、输入话题名、输出 root、发布频率和超时值；不为频率或字段隐式回退到零值。

## 5. 数据采集设计

### 数据内容和格式

上游 `next_record` 使用 `ApproximateTimeSynchronizer` 同步多个 `JointState` 流，默认 target 50 Hz、最低有效频率 45 Hz、容许时间偏差 30 ms。按 `r` 开始/停止 episode，输出 HDF5：

```text
ep_0000/<key>/data
ep_0000/<key>/timestamps
```

每个 episode 必须包含同长度、同顺序的：`joint_pos`、`joint_vel`、`joint_cmd`、`measured_joint_torque`。时间戳用纳秒。采集后运行上游 `check_h5`，检查 schema、行数、采样频率、时间单调性、间隔突增和 NaN/Inf。

### 运动覆盖

只用无接触自由运动构造训练集。建议在软件和机械限位的安全范围内覆盖：

- 多个静态姿态，以及姿态之间的往返运动。
- 慢速、中速和计划工作速度；正反方向和不同加减速过程。
- 单关节主导运动与多关节协调运动。
- 不同工作空间区域、不同负载/末端工具状态（若末端配置会改变惯量，配置需分别记录并评估是否分别训练）。

自由空间训练数据不能包含碰撞、接触、手推关节或线缆拉扯等外力样本，否则模型会把部分外力学进 `tau_free_hat`，导致接触残差变小。温升、摩擦、供电和姿态变化应跨多个采集批次记录，作为泛化检查条件。

使用 `joint_position_controller` 执行确定的轨迹并同时记录实际命令状态。手动拖动采集不适合作为主要数据源，因为控制器目标可能保持、轨迹会与手动操作竞争，且目标—实测误差不能代表规划输入。采集前先用低风险小幅轨迹确认 q、qdot、q_cmd 和 tau 的方向、单位及响应延迟。

### 训练集划分

优先按整次采集 session 或 episode 划分 train/validation；不要先把时间样本随机拆分，因为相邻滑动窗口高度重叠，会造成验证数据泄漏。单文件内的 contiguous split 需保留至少一个 history 长度的 buffer；更稳妥的是独立采集验证文件。记录配置、机械臂侧别、软件 commit、标定文件版本、末端工具、轨迹参数、环境与温度等元数据。

## 6. 模型和训练

上游训练代码将每帧特征拼为：

```text
x_t = concat(q_t, qdot_t, q_cmd_t - q_t)
y_t = tau_measured_t
```

每个历史窗口预测窗口最后时刻的力矩。训练使用均方误差，均值/标准差仅由训练集拟合，并保存供推理复用。默认 `history=50`，LSTM hidden size 128、2 层，回归 head hidden size 256、2 层、dropout 0.1，batch size 2048、Adam 学习率 0.001、默认 20 epoch。单臂 7 关节时每帧输入为 21 维，窗口张量约为 `[50, 21]`，输出为 7 维关节力矩。

初次训练可沿用默认值做 smoke test，但不能以默认 20 epoch 或单一随机种子作为最终结论。最终实验需通过独立验证集比较每关节 RMSE/MSE、误差偏置、不同速度/姿态分组误差以及残差频谱/波动；记录配置和 checkpoint。每侧单独训练和部署一个模型，不在第一阶段合并左右臂，以避免左右机械差异和命名映射混淆。

每次 run 保存：

```text
model.pt
config.yaml
normalization.npz
metrics.json
```

checkpoint 目录命名要关联机械臂侧别、训练数据版本和时间。任何输入维数、joint order、history 或归一化变化都需要新 run，不能复用旧 checkpoint。

## 7. 环境安装、构建与运行

### W3 目标接口阶段

修改 W3 `ieir_controllers` 的控制器头文件与实现，补充 `sensor_msgs`/`realtime_tools` 依赖声明（若已由当前依赖间接提供仍要显式声明使用项），加入不依赖真机的目标插值/发布测试。按现有脚本只构建 W3 工作区；不需要接 CAN 做编译和单元测试。接口完成后 source W3 overlay，使用 mock 或仿真测试话题和时间戳，再在真机上仅观察话题，不改变轨迹命令。

### FACTR2 包阶段

在独立 venv 中安装上游核心依赖和适配器依赖；PyTorch 按是否使用 CUDA 选择官方匹配 wheel。先完成包构建与命令帮助检查，再运行离线 H5 检查、训练和 checkpoint 重载测试。不要运行 `sudo pip`，不要让 venv 覆盖 ROS apt 提供的 Pinocchio 或 ROS Python 模块。

### 真机采集终端分工

1. 终端 A：按你现有流程启动 W3 CAN bridge。
2. 终端 B：按现有流程启动控制端，确认双臂反馈、标定坐标和 controller 状态正常。
3. 终端 C：source Humble 和 W3 overlay，启动 NEXT adapter；检查 `command_state` 与四路输出频率、joint names 和数值。
4. 终端 D：在独立环境启动 `next_record`，采集无接触自由运动 episode。
5. 离线训练时使用 FACTR2 环境读取 H5，不必连接 CAN；推理时回到同时可见 W3 ROS graph 的 NEXT 终端运行。

启动控制器、切换控制模式和发送轨迹仍由现有 W3 流程负责。NEXT recorder、trainer、adapter 和 inference 均不得调用 enable/disable 服务或发布 motor command。

## 8. 离线验证与在线推理

离线评估先在独立无接触验证集运行模型，检查自由力矩预测与实测力矩是否一致，再检查残差：

```text
tau_ext_raw = tau_measured - tau_free_hat
```

没有力传感器时，无接触残差可以用于评估模型能否解释自身动力学，但不能作为外力估计的真实标签。用低风险、可重复的轻微外部接触检查残差是否在接触时增大、离开后回落，并按每关节记录原始与滤波信号；不能据此声称已验证绝对牛顿级末端力精度。

在线 `next_infer` 与训练共享相同四路输入配置，使用滚动 HistoryBuffer。窗口未满前不预测。上游发布自由空间力矩、raw residual、平滑 residual、MSE、normalized contact magnitude 和带迟滞的 contact state。`external_joint_torque` 是平滑值，`external_joint_torque/raw` 是直接差值；两者都应记录用于调参和审计。`contact_state` 的阈值/scale 是项目配置，不是模型自动学出的物理阈值。

第一阶段 inference 独立运行并只发布 NEXT 命名空间话题。不要把原仓库 Piper/Gello teacher-arm feedback demo 接入 W3；其增益、关节数、符号和驱动控制都是特定硬件配置。后续若研究反馈，应作为独立项目阶段，逐关节验证方向和限幅，并重新做控制稳定性与失效处理设计。

## 9. 分阶段实施清单

### Part A：控制目标接口

- 在 `JointPositionController` update 中发布插值后的 `pos_des/vel_des`。
- 使用 realtime publisher，定义消息名序、时间戳、active/inactive 语义。
- 验证轨迹前导、段间插值、轨迹结束保持、部分关节和重新激活行为。

### Part B：W3-NEXT adapter

- 新建独立 `factr2_w3_adapter` ROS 2 Python 包。
- 实现参数化 side/topic/rate/timeout。
- 按名称对齐 q、qdot、q_cmd、tau 并固定为 7 维。
- 验证缺失关节、数组乱序、陈旧消息、非有限值和 QoS。

### Part C：环境与接口验收

- 独立配置 W3 Humble/Python 3.10 与 NEXT venv。
- 验证两个工作区 overlay 不互相污染且同 ROS domain 发现正常。
- 先 mock topic，不接真机，验证 adapter 到 recorder 的数据形状与同步。

### Part D：自由运动采集

- 设计小幅低风险轨迹和运动覆盖表。
- 采集多个 session，按整 episode 分 train/validation。
- 用 H5 检查工具检查频率、丢帧、时间戳和数值。

### Part E：训练与离线评估

- 固定关节顺序、history、采样率与训练配置。
- 训练单侧模型，保存完整 run artifact。
- 在不同 session/轨迹条件下报告每关节误差和残差稳定性。

### Part F：只读真机在线推理

- adapter 与 `next_infer` 并行运行，打开可视化。
- 先做无接触观测，再做轻微、可重复接触验证。
- 评估时记录温度、姿态、速度、接触时刻、原始/滤波 residual 和时延。
- 确认 NEXT 崩溃、话题超时或 checkpoint 不可用时不影响 W3 控制。

## 10. 验收标准

- `command_state` 反映控制器本周期实际目标，而非原始 trajectory 点。
- adapter 对单臂输出固定 7 关节、名称一致、数值有限，持续频率接近 50 Hz；陈旧或无效输入不会被补零伪装成有效样本。
- H5 四流行数一致，时间戳单调，采样频率达到配置要求，没有 NaN/Inf 和显著同步错位。
- 验证数据与训练 episode/session 分离；模型在未参与训练的自由运动轨迹上残差稳定。
- 在线模型输入顺序/缩放与训练一致，窗口达到 50 帧后开始输出，端到端推理不阻塞 W3 控制循环。
- 轻微接触可重复地产生显著于无接触噪声的变化；报告保留其只是关节外力估计、无传感器绝对真值的限制。

## 11. 主要风险及处理

- **真实力矩反馈偏差**：达妙电机估算力矩可能受温度、死区、摩擦和驱动滤波影响。先用现有 `teleop_force_observer` 记录测得力矩、Pinocchio 重力项、摩擦项与 residual，作为质量诊断，不将其当 NEXT 训练标签或外力真值。
- **输入时间错位**：W3 状态、目标和 controller loop 不同频。目标接口应从同周期生成；adapter 统一输出时间戳并设置超时；H5 检查丢帧。
- **动力学覆盖不足**：姿态/速度范围窄会让模型把分布外运动误判成接触。按工作空间、速度和方向分层采集并验证，线上标记大误差/低置信数据而不反馈控制。
- **训练数据污染**：接触混进自由运动集会吞掉外力残差。每个 episode 记录接触情况并人工剔除污染片段。
- **Python ABI 与 ROS 绑定**：Humble 的 rclpy 与 Python 3.12 venv 通常不匹配。优先 Python 3.10 独立 venv；无法满足依赖时使用独立 ROS 容器/环境，绝不覆盖 W3 系统 Python。
- **模型输出被误用**：NEXT residual 不是经过安全认证的闭环信号。初版 adapter/inference 全只读，独立监控进程不能拥有电机命令接口。

## 参考

- FACTR2 NEXT 上游代码：本地 `factr2_next/README.md` 与 `src/factr2_next/factr2_next/`。
- FACTR2 项目页：<https://jasonjzliu.com/factr2/>。
- FACTR 2 论文：<https://arxiv.org/abs/2606.12406>。
- W3 控制接口和坐标约定：`/home/dingyj/w3_dual_arm_ws/src/docs/CONTROL_PIPELINE.md` 与工作区根 `README.md`。
