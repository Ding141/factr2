# 右臂机器人连接与 NEXT 模型启动手册

更新：2026-10-10。当前本机目录：`/home/venom/dual_arm_robot` 与 `/home/venom/factr2`；两个仓库均使用 `liyq-dev`。

## 1. 本次保存了什么

- W3：修复新轨迹首周期跳变、未来轨迹衔接、短窗口结束时突停；五次 Hermite 插值连续衔接位置/速度/加速度，带速度、加速度和 jerk 限制。连续重复中间边界按跟踪误差检查，真正停止终点仍保留到位检查。
- 重力控制器：新鲜规划速度驱动的连续摩擦前馈；力矩限幅 0.8 N·m、变化率 8 N·m/s、参考超时 0.05 s。右臂 J6 试验库仑系数 0.45 N·m × 现有增益 0.8，尚不等于完成摩擦标定。
- FACTR2：平滑覆盖采集、全反馈录制、控制目标连续性审计、严格分段恢复、无共享帧划分、训练与评估、关节残差网页、末端等效六维力/力矩网页。
- 新模型副本：`models/w3/right/c2_20261010/`，保存实际权重、归一化、冻结训练配置、元数据、损失、划分审计、采集诊断和评估摘要。原始模型目录仍保留。
- 两段原始录制分别为 `data/quick_capture/right/20261010_160331_906917/` 和 `data/quick_capture/right/20261010_164151_566506/`。第一段只使用完整通过检查的慢速 J1～J6；未完成的 J7 段排除。第二段完成剩余慢速、全部快速和末端路径。原始记录未修改。
- 有效帧 208858：train 106024、val 60960、test 41874；可用历史窗口分别 105240、60421、41335。80 轮训练，以验证集选择第 78 轮。测试总体关节力矩 RMSE 0.054053 N·m，均值基线 1.393885 N·m。属于同一活动的重复动作留出，不是独立会话/接触精度评估。
- 权重 SHA256：`47e6cb562a77d41955191f994c864e3d7abd3b6bb529ef3faee0449d488606b0`。

## 2. 适用范围和当前限制

仅右臂、七个关节、无夹爪、空载。本机 CAN1 对应右臂，channel=1；CAN0 为左臂。右臂标定文件未更改，SHA256 为 `23532d9890f7e3ece77a80f0af8a26742c6bcf85a1cd07ceda3147dd528e6838`。

模型在电脑 CPU 上运行，通过 ROS 订阅机器人反馈，不烧录到电机。模型及两张观测网页都不发送运动/力矩命令。位置控制和重力补偿由 W3 控制端执行。

统一 `ROS_DOMAIN_ID=74`、`ROS_LOCALHOST_ONLY=1`、`RMW_IMPLEMENTATION=rmw_fastrtps_cpp`。这是同一台电脑连接 CAN 机器人的配置，不是多电脑 ROS 通信配置。所有终端必须一致，避免混入其它域的机器人或仿真。

当前已确认：模型对某些组合姿态仍有明显空载残差，施力后可响应但卸载基线不一定恢复；关节独立范围内不等于所有组合姿态都充分训练。网页 `valid/估计运行中` 仅表示数据和几何门禁通过，不表示已验证物理精度。模型不可作为已标定的末端力传感器或直接用于控制闭环。

## 3. 启动顺序与终端角色

| 终端 | 用途 | 是否使能或控制机器人 |
|---|---|---|
| A | CAN1 通信桥 | bridge 默认不自动使能 |
| B | W3 网页与受管理的右臂控制端 | 打开网页不使能；启动控制端时使能 |
| C | 原始电机反馈健康监测 | 只读 |
| D | 100 Hz 右臂数据适配器 | 只读 |
| E | NEXT 模型与关节力矩网页 8081 | 只读 |
| F | 末端等效力与力矩网页 8082 | 只读 |

顺序：确认支撑与配置 → A → B 网页启动右臂控制、切关节位置模式 → 核对反馈 → C → D → E → F。
已经连接、控制器已正常运行时，保留 A/B，只启动缺失的 C～F；不能重复启动同名 bridge、controller_manager、adapter 或观测服务。

## 4. 启动前检查

冷启动、控制端重启或退出前可靠支撑右臂；左臂保持支撑且不进入右臂活动区。无夹爪时始终 `gripper:=false`。确认机器人电源、急停恢复状态、USB/CAN线缆、CAN1 物理通道；不把视觉接近零位当作标定完成。

以下使用新的系统终端，不激活 Conda 或 FACTR2 venv，不 source 旧 `w3_dual_arm_ws/install`。每个 W3 类终端 A/B/检查终端先执行：

```bash
cd /home/venom/dual_arm_robot
source /opt/ros/humble/setup.bash
source install/local_setup.bash
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 pkg prefix ieir_bringup
ros2 pkg prefix ieir_controllers
```

两个 prefix 应指向 `/home/venom/dual_arm_robot/install/`。校验实际右臂标定：

```bash
sha256sum /home/venom/dual_arm_robot/src/ros2_ws_config/joint_offsets_right.yaml
ip -details -statistics link show can1
```

若 can1 不存在，先解决 USB/CAN 驱动与接口识别；不能用 can0 替代右臂。

## 5. 终端 A：电脑与机器人的 CAN1 连接

若 CAN1 已是 UP 且 FD 1M/5M 配置正确，直接启动 bridge。只有冷启动、控制端已停、右臂可靠支撑、CAN1 处于 DOWN 时执行以下配置；不要对运行中的机器人执行 down/up：

```bash
sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can1 txqueuelen 1000
sudo ip link set can1 up
ip -details -statistics link show can1
```

sudo 密码只在本机终端输入。随后在 A 执行：

```bash
ros2 launch ieir_bringup bridge.launch.py arms:=right gripper:=false
```

保持 A 运行。这里 `arms:=right` 保留 channel=1，不重编号，不启动左臂。CAN 没有 bus-off、持续错误或断流才进入下一步。

## 6. 终端 B：机器人控制网页与右臂控制器

在另一个按第4节配置好的 W3 终端执行：

```bash
ros2 launch ieir_bringup ui.launch.py \
  workspace:=/home/venom/dual_arm_robot port:=8766 gripper:=false
```

打开 http://127.0.0.1:8766/。操作流程：

1. 核对七个右臂电机在线，网页反馈与实物一致。
2. 开启网页控制权限；“运行总览”选择右臂，确认“启动控制端”。这一步会使能并启动右臂重力补偿；默认模式为 gravity。
3. 确认重力补偿正常，在网页切到关节位置模式（joint），使 `joint_position_controller` active；其启动持有当前位置，不自动回到模型参考姿态。当前模型采集/部署使用位置保持+重力补偿，纯重力手推模式不能视为已验证的相同控制条件。
4. 检查下方三个控制器 active 后，再解除临时支撑，保持空载无接触。

在额外的 W3 检查终端执行：

```bash
ros2 control list_controllers
ros2 topic echo /joint_position_controller/command_state --once
ros2 topic echo /joint_states --once
```

期望 `joint_state_broadcaster`、`gravity_compensation_controller`、`joint_position_controller` 均 active。command_state 有 `right_joint_0`～`right_joint_6` 七项；停止保持时目标速度为零。

B网页管理方式与直接启动方式只能选一个。若不用网页启动控制端，可在单独终端执行以下等价的右臂位置控制启动命令；**会立即使能机器人**，要求已支撑，且没有其它控制端：

```bash
ros2 launch ieir_bringup controllers.launch.py \
  arms:=right controller_type:=joint_position enable_gripper:=false \
  offsets_yaml:=/home/venom/dual_arm_robot/src/ros2_ws_config/joint_offsets_right.yaml \
  friction_model_yaml:=/home/venom/dual_arm_robot/src/ros2_ws_config/friction_model.yaml
```

此时 B网页不管理该外部控制进程，其停止由上面的终端负责。不要再从网页点击启动控制端。启动模型无需回零，也不要为了让图线出现直接发送零位或大范围轨迹。

## 7. 终端 C：只读硬件健康器

C 使用系统 ROS 与 W3 消息包：

```bash
cd /home/venom/dual_arm_robot
source /opt/ros/humble/setup.bash
source install/local_setup.bash
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
/usr/bin/python3.10 /home/venom/factr2/factr2_w3_adapter/tools/w3_health_monitor.py
```

real adapter 要求新鲜、完整、正常的右臂原始电机反馈，不能通过关闭健康门禁来隐藏通信错误。

## 8. 终端 D：100 Hz 右臂适配器

D/E/F 都通过 `scripts/next.sh` 使用隔离的 FACTR2 venv，不叠加 W3 overlay：

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
scripts/next.sh ros2 launch factr2_w3_adapter adapter.launch.py \
  side:=right profile:=real publish_hz:=100
```

适配器输出 `/factr2/right/joint_pos`、`joint_vel`、`joint_cmd`、`joint_effort` 和 `status`。启动初期可短暂等待健康器/输入，持续异常则先排查 A/B/C。

## 9. 终端 E：启动本次模型及关节力矩网页

本次固定配置 `config/w3/right/inference_smooth_100hz.yaml`；默认 checkpoint 为仓库内副本 `models/w3/right/c2_20261010`，`next.sh` 自动从仓库根目录启动。训练配置内部原始数据路径是来源记录，加载推理不需要存在这些 H5。

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
scripts/next.sh ros2 launch factr2_next readonly.launch.py \
  side:=right \
  inference_config:=/home/venom/factr2/config/w3/right/inference_smooth_100hz.yaml \
  web_config:=/home/venom/factr2/config/w3/right/visualize.yaml
```

打开 http://127.0.0.1:8081/。100 Hz × 50 帧约0.5秒预热，默认 CPU 单线程。特征为 `[q, qdot, q_cmd-q]`，每帧21维；输出七维自由空间力矩。滤波 EMA alpha=0.2，不做在线自动归零。

默认红线是七关节滤波残差的 L2 范数，绿线是 L1 范数/scale；本配置 scale=1，单位 N·m。可勾选逐关节 Filtered/Raw/Free prediction。横轴为该网页节点运行时间，单位秒。MSE 单位 (N·m)²，contact 为无单位0/1。

## 10. 终端 F：末端等效力与力矩网页

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
scripts/next.sh python scripts/w3_endpoint_wrench.py \
  --config config/w3/right/endpoint_wrench.yaml
```

打开 http://127.0.0.1:8082/。上图 Mx/My/Mz，单位 N·m；下图 Fx/Fy/Fz，单位 N；横轴为节点运行秒数。红/绿/蓝分别表示 X/Y/Z。作用点和坐标轴为 `right_attachment_point`（右臂末端安装点），不是底座原点。

换算 `tau_residual ≈ J(q).T @ [F; M]`，使用缩放阻尼最小二乘；假定全部外部接触发生在末端。臂杆/多点接触、模型误差和摩擦同样可能投影成末端量。方向符号尚需已知方向加载验证。

## 11. 模型工作范围与末端空白原因

| 关节 | 采集目标范围（°） |
|---|---|
| J1 | -135～105 |
| J2 | -0.5～155.4 |
| J3 | -115.9～115.9 |
| J4 | -99～69 |
| J5 | -115.9～115.9 |
| J6 | -31.9～31.9 |
| J7 | -72～72 |

约机械跨度80%，不是机械硬限位认证。采集慢/快上限6/20°每秒，末端XYZ幅度/圆半径20mm。范围检查允许3°跟踪余量；该余量不是额外允许自动运动的限位。

末端显示要求：机器人描述SHA匹配、状态与残差精确同时间戳且新鲜、NEXT valid、范围门禁通过、缩放雅可比条件数≤100、相对拟合残差≤0.35。任何失败都会留空并提示原因，空白不代表零外力。

参考采集中心 `[0,50,0,-55,0,0,0]°` 曾具有较好几何条件；这只是记录，不是启动自动回位命令。接近全伸直零位可能奇异，不能通过放宽门限强行画曲线。

## 12. 只读运行检查与常见问题

W3类检查终端：

```bash
ros2 topic info /joint_position_controller/command_state --verbose
ros2 topic hz /joint_position_controller/command_state
```

NEXT类检查终端：

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1
scripts/next.sh ros2 topic echo /next/right/status --once
scripts/next.sh ros2 topic echo /next/right/endpoint_wrench/status --once
curl --fail --silent http://127.0.0.1:8082/snapshot
```

NEXT status应显示valid、history_count=50、output_hz约100，source_age小于0.25秒；推理耗时随CPU负载变化。两个模型/网页状态不能代替现场接触和精度确认。

- 找不到topic：核对domain74、localhost1、启动C/D、以及command_state控制器是否active。命令终端也须相同域。
- `health_missing_stale_or_error`：核对C和电机新鲜反馈；电机状态码1在当前协议表示已使能正常，不是故障1。
- `warming`：等待完整50帧；持续发生时检查断流/时钟/健康状态。
- `out_of_scope`：范围外；关节残差可继续显示，但接触报警被抑制。不能据contact=false推断没有外力。
- 8081/8082占用：检查是否已有相同服务，直接使用原网页，不能重复启动。
- 末端空白：看8082状态说明，优先查奇异姿态、输入状态、训练范围、描述hash或末端接触拟合检查。
- 空载出现非零力：先检查关节原始残差与自由力矩预测。当前组合姿态已有明显空载偏差，需要独立空载验证/补采；单个关节范围内不保证组合覆盖。不要靠自动扣零隐藏模型误差。
- 停止发送轨迹后仍短暂运动：已发窗口及连续制动尾段可继续执行，软件停录不是急停。
- 模型profile/hash失败：不要删门禁；核对完整checkpoint五个必需文件、100Hz/右臂/50帧及标定条件。

## 13. 停止与重新连接

只停止观测：F → E → D → C 的各终端按Ctrl+C。A/B保持运行，机器人控制不因此停止。关闭网页标签不停止ROS观测进程。

需要重启机器人控制/连接：先可靠支撑右臂；停止观测；网页管理模式下从B停止其自有控制端，再停止B，再停止A；直接控制模式则从对应控制终端Ctrl+C。CAN看门狗的阻尼不能托住重力，退出GC前必须支撑。需要重新配置CAN时，只在控制端/bridge停止后操作。随后按A→B→检查→C～F重新启动。不能用广泛pkill杀掉所有ROS进程。

网页管理模式下Ctrl+C退出B会停止其自有控制端；仅关闭浏览器标签通常不会停止控制端。停止A会断开反馈/命令链路；不能在无支撑时随意退出。

## 14. 源码、环境更新与构建

本机现有环境已构建，日常启动无需重复构建。修改C++控制器后，离线测试通过仍不会让运行进程自动换库；支撑、停控制、构建/安装、重启后才生效。运行中不要覆盖正在使用的控制器库。

W3构建命令（停止控制后）：

```bash
cd /home/venom/dual_arm_robot
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select ieir_controllers ieir_bringup
source install/local_setup.bash
```

FACTR2构建命令（已有本机venv，停C～F后）：

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1
scripts/next.sh colcon build --symlink-install --packages-select factr2_next factr2_w3_adapter
```

全新电脑的系统依赖/venv安装见 `docs/environment.md`、W3 `src/README.md`；模型权重已入库，但本机标定不能用别台机器的文件替代。目录改变时修改 endpoint_wrench.yaml 的 inference_config 绝对路径及启动参数，核验模型/标定条件；不要修改冻结checkpoint中的config.yaml或metadata.json来伪造相同profile。

## 15. 备份、结果与后续采集

本机七个零偏/方向/限位reviewed标定文件另备份在 `/home/venom/dual_arm_robot/log/backups/work_20261010/`，manifest.json记录SHA。标定不提交Git；从远程新克隆仓库时须恢复本机备份，并核对SHA与实际装配。

完整模型原目录：`/home/venom/factr2/runs/next_right_c2_campaign_20261010_164151_566506_session_trial_20261010_174837_628319/`；完整采集、bag、训练CSV、启动日志保留在原data/runs/log路径，Git只提交小checkpoint及摘要。`models/w3/right/c2_20261010/ARCHIVE_MANIFEST.json`给出文件hash和原始来源。

后续补采不能直接重播历史冻结计划：当前起点可能不同。必须重新规划、碰撞检查、核对净空并获得现场开始许可。启动模型不包含任何采集或运动动作。本次平滑采集和恢复流程见 `RIGHT_SMOOTH_CAPTURE.md`；独立空载/已知接触验证仍待完成。

论文明确用自由运动学习内部力矩与50帧历史，未专门给出卸载自动归零；关于多姿态静止补采是本机诊断建议。原文：https://arxiv.org/html/2606.12406v1 。
