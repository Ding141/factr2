# FACTR2 + W3 真机调试逐步执行指南

核对日期：2026-10-09（Asia/Shanghai）。依据当前本地源码：FACTR2 `eb0240bfbb93b6125cc8d40a40d5b27b6144566a`，W3 `d32262d66b4f94ce2105eddcd3bea1653ce34b42`。本文是操作说明，不是真机验收报告；编写时没有启动硬件节点、使能电机或发送运动目标。

## 1. 当前处于哪一步

现有第 1–6 阶段报告均为离线验收。检查到 NEXT 环境为 Python 3.10、CPU PyTorch，五个核心 ROS 入口及 adapter 入口已安装，左右配置模板结构检查通过。当前检查到的 14 份数据侧车、2 份模型 metadata 的 `source` 均为 `synthetic`，尚未找到真实数据/真实合格模型。

本机左右 offsets 各 7 项、`complete: true`；dual 的 14 项 name/channel/slot/offset/sign 与两臂文件一致。然而 dual 的 14 项 `visually_verified` 均不是 true，不能据文件完整性宣称实物姿态已核对。不要直接修改此字段冒充验证；在现场完成实物核对并记录结果。

本次只读系统检查返回 `Device "can0" does not exist`、`Device "can1" does not exist`。这是检查时此环境的状态；先接好适配器、确认驱动与 SocketCAN 接口，不能据此判定机械臂本身故障，也不能在接口不存在时继续启动 bridge。

主线：检查环境与 CAN → 核对标定 → W3 启动与小范围验证 → 健康器/real adapter → pilot 录制 → 数据质量 → 独立 train/val/test → 左右模型训练 → 冻结预算/评估 → 单侧只读推理 → 无接触/接触验收 → 双侧 → 收尾。

第一次只完成到“pilot 数据质量通过”即可。这是有明确产物的首次调试目标；完整复现还需要后面的真实模型和接触验证。

## 2. 先理解两个系统

| 对象 | 工作 | 会不会驱动机器人 |
|---|---|---|
| W3 CAN bridge | 原始电机反馈、命令通信 | 是命令链路；启动 bridge 本身不自动使能 |
| W3 控制端 | 标定转换、重力/位置控制 | 启动会使能所选电机 |
| W3 网页 8766 | 显示状态、管理控制端、确认目标、示教回放 | 打开页面不使能；确认控制/回放动作会驱动 |
| W3 健康器 | raw 电机反馈转标准诊断 | 只读 |
| NEXT real adapter | 选侧别、按名字重排、检查时效/健康、输出 50 Hz 四流 | 只读 |
| NEXT recorder | 四流写 H5 | 只录数据 |
| NEXT trainer/evaluator | 训练/验证自由运动力矩模型 | 离线，无需连接机器人 |
| NEXT inference/网页 8080/8081 | 自由力矩预测、残差、接触指示 | 只读，不提供 W3 失效停车联锁 |

模型输入每帧为 `[q, qdot, q_cmd-q]`，21 维，历史 50 帧；输出七维自由力矩预测。残差为实测电机反馈力矩减自由力矩预测，单位 Nm，包含建模误差与噪声。它不是已经标定的末端六维力。

| 信号 | 来源/输出字段 | 单位 |
|---|---|---|
| q | `/joint_states.position` → `/factr2/left/joint_pos.position` | rad |
| qdot | `/joint_states.velocity` → `/factr2/left/joint_vel.velocity` | rad/s |
| q_cmd | `/joint_position_controller/command_state.position` → `/factr2/left/joint_cmd.position` | rad |
| tau | `/joint_states.effort` → `/factr2/left/joint_effort.effort` | Nm |

`/joint_position_command` 是轨迹输入，不能替代 q_cmd。控制器的 command_state 才是本周期插值后实际写入接口的目标。position inactive 时不持续发布该目标，所以纯重力模式用于示教/反馈检查，NEXT 自由运动采集和推理需 position active。

本文按已有操作文档的“同机 Humble、无夹爪、空载、双臂控制端，先只运动左侧”展开。若实物已装夹爪、负载改变或只启动单臂，必须同步改变 W3 配置及数据 profile；右臂独立使用 right 数据与模型。启动双臂即使只移动左侧，也会使能双侧，需支撑全臂。

## 3. 新终端如何准备

长期运行进程分别占一个终端，观察命令另开终端。不要把下文长期进程逐行粘到同一个 shell；上一条不会退出，后续命令不会执行。

| 名称 | 环境 | 运行内容 |
|---|---|---|
| W1 | W3 | 唯一 bridge |
| W2 | W3 | 唯一管理 UI，控制端作为其子进程 |
| W3 | W3 | 健康器 |
| W4 | W3 | rosbag 审计 |
| W5 | W3 | 状态检查、参数快照；需要时另开示教/回放终端 |
| N1 | NEXT | 左侧 real adapter |
| N2 | NEXT | 左侧 recorder，随后关闭 |
| N3 | NEXT | 工具/服务/质量/训练/评估 |
| N4 | NEXT | 合格模型的 readonly inference，后期才用 |

### 3.1 每个 W3 终端：都从新终端执行

```bash
cd /home/dingyj/w3_dual_arm_ws
mkdir -p log/ros log/tmp
env -i HOME="$HOME" USER="${USER:-dingyj}" PATH=/usr/bin:/bin LANG=C.UTF-8 \
  PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=73 ROS_LOCALHOST_ONLY=1 \
  RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOG_DIR="$PWD/log/ros" TMPDIR="$PWD/log/tmp" \
  FASTRTPS_DEFAULT_PROFILES_FILE=/home/dingyj/factr2/config/w3/dds_loopback.xml \
  bash --noprofile --norc
source /opt/ros/humble/setup.bash
source install/local_setup.bash
export ROS_LOCALHOST_ONLY=1
```

`env ... bash` 会进入新的干净交互 shell，随后再执行 source。退出这个 shell 用 `exit`。

```bash
command -v python3
ros2 pkg prefix ieir_controllers
ros2 pkg prefix ieir_bringup
ros2 pkg prefix w3_robot_bridge
```

应为系统 Python，以及 `/home/dingyj/w3_dual_arm_ws/install/...`。若加载旧工作区或 NEXT venv，关闭终端重新开始。

### 3.2 每个 NEXT 终端

```bash
cd /home/dingyj/factr2
export ROS_DOMAIN_ID=73 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/config/w3/dds_loopback.xml"
scripts/next.sh python scripts/environment_probe.py
scripts/next.sh ros2 pkg executables factr2_next
scripts/next.sh ros2 pkg executables factr2_w3_adapter
scripts/next.sh python scripts/check_w3_configs.py
```

应看到 `.venv/bin/python`、只有 FACTR2/Humble overlay，以及 `next_record/next_train/next_infer/next_check_h5/next_visualize` 和 `w3_next_adapter`。模板 PASS 不等于已填写现场路径；正式启动仍需 `--runtime` 校验。

两种终端使用相同 domain 73 和同机 DDS XML。已有 W3 正在别的 domain 运行时，先确认实际 domain，让所有观察节点跟随，不为 NEXT 接入贸然重启运动中的控制端。同机 XML 只允许 loopback；跨机方案不能照抄。

当前已有环境/构建，不必重新安装。仅缺失或改代码时：

```bash
# W3：只检查依赖；首次缺依赖再按 W3 README 安装
bash src/scripts/install_dependencies.sh --check
# W3：改 C++/包内配置/launch 或首次构建
bash src/scripts/build_workspace.sh "$PWD"
```

```bash
# NEXT：仅环境缺失时
bash scripts/setup_next_env.sh
# NEXT：只构建本流程使用的包
scripts/next.sh python -m colcon build \
  --base-paths factr2_next/src/factr2_next factr2_w3_adapter --symlink-install
```

构建后重新准备运行终端。W3 用系统 ROS/Pinocchio，NEXT 用独立 venv；不要把两个 install source 到同一 shell，不启动上游 Piper/Gello/teacher-arm bringup。

## 4. W3 启动前：CAN 与标定

### 4.1 W5：只读看 CAN

```bash
ip -br link
ip -details -statistics link show can0
ip -details -statistics link show can1
```

can0=左臂/channel0，can1=右臂/channel1，每侧 slot0..6 对应 CAN ID1..7。正常应接口存在且 UP，CAN FD 参数与本机硬件匹配，无持续 bus-off/错误增长。

接口不存在时先查适配器是否连接、SocketCAN 驱动/固件是否正确、接口是否换名、物理通道是否调换；`ip link set` 不能创建缺失的适配器驱动。可只读查看 USB 设备：

```bash
lsusb
```

仅当接口已存在且确认使用仓库的 1 Mbps/5 Mbps CAN FD 配置，控制已经停止、全臂有支撑时，才执行以下系统配置；这不是每天都必须做的步骤：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up
sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can1 txqueuelen 1000
sudo ip link set can1 up
ip -details -statistics link show can0
ip -details -statistics link show can1
```

CAN 参数不匹配、接口不 UP、错误持续增长时停止后续流程。不要在控制运行时重配总线，也不要启动旧 USB2CAN。

### 4.2 标定检查

```bash
ls -l src/ros2_ws_config/joint_offsets_left.yaml \
  src/ros2_ws_config/joint_offsets_right.yaml \
  src/ros2_ws_config/joint_offsets_dual.yaml \
  src/ros2_ws_config/friction_model.yaml
```

现有文件已结构核对通过。实物未更换电机/装配/电机零位时先保留现有标定，随后现场核对姿态；不要为了“走流程”重新标定或自动回零。标定异常才回到 W3 [完整标定手册](/home/dingyj/w3_dual_arm_ws/src/README.md)，先停止控制端、遥操作、回放，仅保留 bridge，并支撑全臂。

需要重新标定的分支命令如下，不属于每次运行的必做命令：

```bash
# W3 终端；先完成上述停止/支撑条件；仅在确需重标时使用
C="$PWD/src/ros2_ws_config"
mkdir -p log/calibration_backups
cp -a "$C" "$PWD/log/calibration_backups/config_$(date +%Y%m%d_%H%M%S)"
SIDE=left
ros2 run ieir_controllers joint_zero_calibration --mode direction \
  --calibration-yaml "$C/joint_calibration_dual_${SIDE}.yaml" \
  --output "$C/joint_directions_${SIDE}.yaml"
ros2 run ieir_controllers joint_zero_calibration --mode limit-preview \
  --calibration-yaml "$C/joint_calibration_dual_${SIDE}.yaml" \
  --output "$C/joint_calibration_dual_${SIDE}_reviewed.yaml"
ros2 run ieir_controllers joint_manual_calibration \
  --calibration-yaml "$C/joint_calibration_dual_${SIDE}_reviewed.yaml" \
  --directions-yaml "$C/joint_directions_${SIDE}.yaml" \
  --output "$C/joint_offsets_${SIDE}.yaml"
# 右侧 SIDE=right 重复，全部完成后才合并
python3 "$C/merge_offsets.py"
```

方向阶段会使能整侧且无重力托举；限位参考只显示；手动采样按 7→6→5→3→4→2→1，首次确认使能当前关节，手推到确认过的参考限位、停稳后确认采样，下一项前失能当前关节。默认网页预览需已有 W3 UI，并保持标定页可见；不需要 MuJoCo。已有正确 directions/reviewed 可按手册跳过对应前置阶段。合并会覆盖 dual 文件；重标后重新合并并重启控制端，无需编译。

## 5. 启动 W3，先验证很小的运动

### 5.1 W1：bridge 长期运行

```bash
ros2 launch ieir_bringup bridge.launch.py arms:=dual gripper:=false
```

唯一 bridge，不自动使能。使能前没有完整有效反馈可能是正常状态。

### 5.2 W2：管理网页长期运行

```bash
ros2 launch ieir_bringup ui.launch.py workspace:="$PWD" gripper:=false
```

打开 [W3 控制台](http://127.0.0.1:8766)。此时只是 UI，不使能。核对运行配置、双臂/夹爪选择、标定路径和实物状态；全臂支撑、急停可用、运动空间确认后，在网页打开“允许控制”→选“双臂”→确认“启动控制端”。**此动作使能双臂并进入重力模式。** 重力模式不等于位置锁定，发现自运动、明显下坠、抖动或姿态错误时先按现场停机流程处理。

已有外部控制端时不要再启动第二套；查其原终端和归属。UI 只管理自己启动的进程。

### 5.3 W5：读状态

```bash
ros2 node list
ros2 control list_controllers
ros2 topic echo /w3_robot_bridge_node/state --qos-reliability best_effort --once
ros2 topic echo /joint_states --qos-reliability best_effort --once
```

初始通常 jsb/gravity active，position inactive。核对 q 的方向与姿态、velocity、effort、时间戳推进，以及 raw 的 online/enabled/error_flags。这里电机已使能正常编码为 `error_flags=1`，不是 0。

网页切“关节位置”，再查：

```bash
ros2 control list_controllers
ros2 topic echo /joint_position_controller/command_state --qos-reliability best_effort --once
ros2 topic info /joint_position_command --verbose
ros2 topic info /joint_position_controller/command_state --verbose
```

采集条件：joint_state_broadcaster/gravity_compensation_controller/joint_position_controller active；双臂 cartesian inactive；没有其他未预期运动目标发布者。检查 command_state 的 name/position/velocity 和推进的源 stamp。

### 5.4 网页小范围验证

“关节控制”选择左臂→“读取当前”→只改一个确认有净空的关节→选较长插值时间→预览并确认发送→观察实物和 q。例如现场确认允许时，可先用当前值相对变化 1°、插值 5 s 做 pilot；这只是小幅示例，不是任意关节的安全保证。关节、方向和幅度由现场确认，避免临近限位/奇异姿态。

网页单位为度，ROS 为 rad；J1..J7 对应 joint_0..6。第一次不要用回零、大范围双臂回放或手写固定 JointTrajectory 练习。发送成功只代表发布，不证明路径安全或已经到达。

## 6. 健康器与 real adapter

### 6.1 W3：健康器长期运行

```bash
/usr/bin/python3.10 /home/dingyj/factr2/factr2_w3_adapter/tools/w3_health_monitor.py
```

### 6.2 W5：看诊断

```bash
ros2 topic echo /factr2/w3_health --qos-reliability best_effort --once
```

对应侧 `factr2/w3/left` 应为 level 0、healthy。需 channel0 的 slot0..6 齐全唯一、online/enabled 为 true、error_flags=1、原始反馈未过期。健康器基于 bridge 报告，不能证明全部 online=true 的数值未冻结。

### 6.3 N1：左侧适配器长期运行

```bash
scripts/next.sh ros2 launch factr2_w3_adapter adapter.launch.py side:=left profile:=real
```

### 6.4 N3：检查四流

```bash
scripts/next.sh ros2 topic echo /factr2/left/adapter_status --qos-reliability best_effort --once
scripts/next.sh ros2 topic echo /factr2/left/joint_pos --qos-reliability best_effort --once
scripts/next.sh ros2 topic echo /factr2/left/joint_vel --qos-reliability best_effort --once
scripts/next.sh ros2 topic echo /factr2/left/joint_cmd --qos-reliability best_effort --once
scripts/next.sh ros2 topic echo /factr2/left/joint_effort --qos-reliability best_effort --once
scripts/next.sh ros2 topic hz /factr2/left/joint_pos --window 100
```

hz 观察足够样本后 Ctrl-C。应七个 left_joint_0..6，字段按第 2 节表，约 50 Hz，adapter level0/reason=ok。四流共享同一选定状态时间戳；不同的 `echo --once` 可能采到不同帧，不能因打印 stamp 不同就认定同步失败，最终用 H5 检查。

adapter 两源 source/receive 年龄≤0.25 s、源间 skew≤0.03 s；无效/陈旧/健康异常就停四流。不要改 mock 或关闭健康门禁来消除报错。

## 7. 建一个真实 session，保存现场身份

### 7.1 N3：建立目录和冻结 profile

```bash
DATASET_DIR=/home/dingyj/factr2/data/left/real_empty_v1
SESSION_DIR="$DATASET_DIR/left_pilot_01"
mkdir -p "$DATASET_DIR/profile" "$SESSION_DIR/audit"
cp -n config/w3/left/record.yaml "$SESSION_DIR/record.yaml"
cp -n /home/dingyj/w3_dual_arm_ws/src/ros2_ws_config/joint_offsets_dual.yaml \
  "$DATASET_DIR/profile/calibration.yaml"
cp -n /home/dingyj/w3_dual_arm_ws/src/ros2_ws_config/friction_model.yaml \
  "$DATASET_DIR/profile/friction_model.yaml"
```

这里按双臂控制端的实际 dual 标定。profile 创建一次后冻结；后续同侧 session 共用同一路径，不能每次复制标定到不同路径。单臂控制端则冻结它实际用的单臂文件，建立对应 profile。`cp -n` 不覆盖已有文件，必须再核对哈希。

所有 shell 变量只在本终端有效。W4/W5/N2/N3 后续使用路径时各自重复赋值；换 session 也要同步改变。

### 7.2 W5：保存实际参数

```bash
DATASET_DIR=/home/dingyj/factr2/data/left/real_empty_v1
SESSION_DIR="$DATASET_DIR/left_pilot_01"
ros2 control list_controllers > "$SESSION_DIR/audit/controllers.txt"
ros2 param dump /controller_manager > "$SESSION_DIR/audit/manager_params.yaml"
ros2 param dump /joint_position_controller > "$SESSION_DIR/audit/position_params.yaml"
ros2 param dump /gravity_compensation_controller > "$SESSION_DIR/audit/gravity_params.yaml"
ros2 param dump /robot_state_publisher > "$SESSION_DIR/audit/robot_description_params.yaml"
sha256sum /home/dingyj/w3_dual_arm_ws/src/ros2_ws_config/joint_offsets_dual.yaml \
  "$DATASET_DIR/profile/calibration.yaml" \
  /home/dingyj/w3_dual_arm_ws/src/ros2_ws_config/friction_model.yaml \
  "$DATASET_DIR/profile/friction_model.yaml" > "$SESSION_DIR/audit/profile.sha256"
```

核对文件非空、各原件/副本哈希相同；保存控制端日志中的实际标定路径。实际 KP/KD 启动时会由 `teleop_joint_gains.yaml` 所选 profile 覆盖，即使 teleop 节点没有启动也不能只抄 `dual_arm_controllers.yaml`。以 ROS 实际参数为准。增益、前馈、URDF、负载变化后旧 profile/模型不自动适用。

### 7.3 N3：填写 record.yaml

可用编辑器打开 `$SESSION_DIR/record.yaml`，保留模板完整内容，仅将 output/session 和 metadata 填为现场实情。以下脚本是无夹爪、空载、确认自由运动时的完整模板填写方式；tool/load/trajectory/温升描述需与现场相符。它只写配置，不启动机器人。

```bash
scripts/next.sh python - "$SESSION_DIR" "$DATASET_DIR" <<'PY'
from pathlib import Path
import sys, yaml
session, dataset = map(lambda x: Path(x).resolve(), sys.argv[1:])
cfg = yaml.safe_load((session / 'record.yaml').read_text())
cfg['output_dir'] = str(session)
cfg['session_name'] = session.name
cfg['metadata'] = {
    'source': 'real', 'tool': 'bare_attachment_v1', 'gripper': 'absent', 'load': 'empty_v1',
    'calibration': {'path': str(dataset / 'profile/calibration.yaml')},
    'control': {
        'gains': {'position_params': str(session / 'audit/position_params.yaml')},
        'feedforward': {
            'gravity_params': str(session / 'audit/gravity_params.yaml'),
            'friction_model': str(dataset / 'profile/friction_model.yaml')}},
    'trajectory_id': 'left_pilot_small_motion_v1',
    'contact': {'present': False, 'label': 'verified_free_motion'},
    'temperature': {'source': 'unknown_no_sensor', 'condition': 'cold_start', 'warmup_minutes': 0},
    'audit_paths': [str(session / 'audit/bag')], 'health_gate_enabled': True}
(session / 'record.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
scripts/next.sh python scripts/check_w3_configs.py \
  --runtime record --config "$SESSION_DIR/record.yaml"
```

应退出 0。此校验不证明 metadata 陈述真实，也不保证健康器已经运行；启动时 recorder 还会读取标定并计算 SHA256。手扶/推压/碰撞/线缆牵拉均不属于自由运动。无传感器就记录未知温度和实际预热时长。

## 8. Pilot：录 bag 和 H5，再操作运动

### 8.1 W4：审计 bag 长期运行

```bash
DATASET_DIR=/home/dingyj/factr2/data/left/real_empty_v1
SESSION_DIR="$DATASET_DIR/left_pilot_01"
ros2 bag record -o "$SESSION_DIR/audit/bag" \
  /w3_robot_bridge_node/state /joint_states /joint_position_controller/command_state \
  /factr2/w3_health /factr2/left/adapter_status \
  /factr2/left/joint_pos /factr2/left/joint_vel \
  /factr2/left/joint_cmd /factr2/left/joint_effort
```

bag 目录应不存在，不复用旧目录。原始 W3 自定义消息要在 W3 环境录/解码。

### 8.2 N2：启动 recorder，尚未开始采集

```bash
SESSION_DIR=/home/dingyj/factr2/data/left/real_empty_v1/left_pilot_01
scripts/next.sh ros2 run factr2_next next_record --ros-args \
  -p config_file:="$SESSION_DIR/record.yaml" -r __node:=next_recorder_left
```

应显示 `Press 'r'`，无 metadata/路径错误。NEXT 普通 YAML 用 config_file；不要传 `--params-file`。启动 recorder 和启动录制是两件事。

### 8.3 操作顺序

1. 位置模式、四流和健康正常，无其他目标源；实物无人工接触。
2. 在 N2 按 `r`，看到 `Writing: ...h5` 与 `Started recording ep_...`，记下实际完整文件名；或在 N3 调用服务。
3. 在 W3 网页执行已确认的小范围运动，包含保持、往返和不同缓慢运动过程。首次可录 30–60 s 检查链路，这个时长不是训练数据充分性的验收标准。
4. 运动结束，N2 按 `r` 停录；或服务 false。然后 N2 Ctrl-C 正常关闭 recorder，等保存完成。
5. W4 Ctrl-C 结束 bag。H5 和 bag 都关闭后再做检查。

```bash
# N3：服务仅控制录制，不控制机械臂
scripts/next.sh ros2 service call /factr2/left/record std_srvs/srv/SetBool '{data: true}'
# 结束本段
scripts/next.sh ros2 service call /factr2/left/record std_srvs/srv/SetBool '{data: false}'
```

服务需 `success: true`；开始被拒绝时查 fresh frame 和发布者，不反复按键。`r` 停录只 flush，Ctrl-C 正常 close 后才生成最终 `.metadata.json` 和 H5 哈希。不要 kill -9。

同一 recorder 每次开录建立 episode；坏帧/断流/回退会分段，不补零、不跨缺口拼窗口。发生人工接触时停止自由运动录制并记录污染；当前接触标签是文件级，无法确认全文件干净就整文件排除并换 session。

## 9. 关闭后检查数据

### N3：H5 严格检查

```bash
DATASET_DIR=/home/dingyj/factr2/data/left/real_empty_v1
SESSION_DIR="$DATASET_DIR/left_pilot_01"
# 先列实际文件；把 H5 赋值为 recorder 打印的已关闭文件，不能填目录
ls -lh "$SESSION_DIR"/*.h5 "$SESSION_DIR"/*.metadata.json
H5=/home/dingyj/factr2/data/left/real_empty_v1/left_pilot_01/SET_ACTUAL_SAVED_FILE.h5
scripts/next.sh ros2 run factr2_next next_check_h5 check "$H5" \
  --json "$SESSION_DIR/quality.json"
```

`SET_*` 必须替换成实际文件名。通过条件：退出 0、accepted=true、四流每行七维、同 int64 源时间、有限数值、时间严格递增、45–55 Hz、最大间隔≤40 ms、有效段≥50 行。≥50 行只是可形成窗口，不代表足够训练。

### W5：bag 检查

```bash
SESSION_DIR=/home/dingyj/factr2/data/left/real_empty_v1/left_pilot_01
ros2 bag info "$SESSION_DIR/audit/bag"
```

预期原始/目标/健康/四流均有消息、时长合理。结合 audit 逐值核对单位、方向、q_cmd/实测响应。不要在真机 domain 用 bag play 重放 `/joint_states` 等来源，会污染正在运行的输入；离线回放应独立 domain。

失败处理：短段排除；gap/非有限/时间异常查 upstream/负载和 audit；hash 失败查文件是否在关闭后变化；metadata 缺失检查是否正常 close。保留原始数据，不手改哈希或删坏行拼连续窗口。

## 10. 从 pilot 扩大为正式采集

每侧至少三个独立真实 session 用于 train/val/test，记录覆盖表：姿态区、单/多关节、正反向、低/中/工作速度、加减速/保持、冷/热条件、无接触确认、排除理由及审计路径。pilot 可进 train/val，不兼作最终独立 test。时长由覆盖及 val 表现决定。

重复第 7–9 节，每次改 session_name/output_dir/trajectory_id/audit_paths/现场条件。保持同侧 train/val/test 的工具、夹爪、负载、标定 profile 和控制设置一致。左/右数据独立，右侧全部使用 right 模板与名称。

### 10.1 想要可重复运动：先示教轨迹，再无人手接触回放录 H5

NEXT 停录→网页切重力模式、position inactive→手拖记录 W3 YAML→停止示教→移开手→网页切位置模式→确认 real 四流恢复→开 NEXT H5→回放。

```bash
# W3 终端：只记录左侧七个关节的手拖轨迹，不是 NEXT 自由运动训练数据
mkdir -p recordings
ros2 run ieir_controllers teach_replay record \
  --output "$PWD/recordings/left_teach_01.yaml" --rate 50 --max-duration 120 \
  --joints left_joint_0 left_joint_1 left_joint_2 left_joint_3 \
  left_joint_4 left_joint_5 left_joint_6
```

完成后 q 或 Ctrl-C 保存。核对全部起点、路径、终点和扫掠空间，做好自由录制准备后：

```bash
# W3 终端：这条会驱动机器人，不与网页回放/遥操作同时启动
ros2 launch ieir_bringup replay.launch.py \
  input:="$PWD/recordings/left_teach_01.yaml" \
  time_scale:=0.5 ramp_in:=5.0 publish_rate:=50
```

进入起点也会运动。回放退出只退回其自行激活的位置控制器；若回放前 position 已 active，退出后可保持 active，执行 `ros2 control list_controllers` 确认。`teach_replay --dry-run` 在判断不发布前会自动切控制器，不能当无副作用预检。网页示教可能录双臂十四关节，回放执行文件中所有关节；不能因文件名含 left 就当单臂轨迹。

## 11. 固定 train/val/test 清单并训练

### 11.1 N3：建立 manifest

```bash
DATASET_DIR=/home/dingyj/factr2/data/left/real_empty_v1
H5_TRAIN=/home/dingyj/factr2/data/left/real_empty_v1/SET_CLOSED_ACCEPTED_TRAIN.h5
H5_VAL=/home/dingyj/factr2/data/left/real_empty_v1/SET_CLOSED_ACCEPTED_VAL.h5
H5_TEST=/home/dingyj/factr2/data/left/real_empty_v1/SET_CLOSED_ACCEPTED_TEST.h5
scripts/next.sh ros2 run factr2_next next_check_h5 manifest \
  --dataset-id left_real_empty_v1 --output "$DATASET_DIR/split.json" \
  --train "$H5_TRAIN" --val "$H5_VAL" --test "$H5_TEST"
scripts/next.sh ros2 run factr2_next next_check_h5 verify-manifest "$DATASET_DIR/split.json"
```

每个 split 后可列多个文件。通过应 PASS，三份非空，来源 `(H5 SHA256, episode)` 无重叠；复制改名不会变成独立数据。操作者仍需保证三个真实独立 session。不要随机拆重叠窗口。contact.present 必须为 false 且真实确认无接触，接触污染文件不纳入 manifest。

### 11.2 N3：生成并校验三份训练配置

```bash
cp -n config/w3/left/train.yaml "$DATASET_DIR/train.base.yaml"
scripts/next.sh python - "$DATASET_DIR" <<'PY'
import copy, sys, yaml
from pathlib import Path
p = Path(sys.argv[1]).resolve()
base = yaml.safe_load((p / 'train.base.yaml').read_text())
base['data']['manifest'] = str(p / 'split.json')
base['save']['output_dir'] = '/home/dingyj/factr2/runs'
base['save']['run_name'] = f"next_{base['side']}_{p.name}"
(p / 'train.base.yaml').write_text(yaml.safe_dump(base, sort_keys=False))
for seed in (0, 1, 2):
    c = copy.deepcopy(base)
    c['train']['seed'] = seed
    c['save']['run_name'] = f"next_{c['side']}_{p.name}_seed{seed}"
    (p / f'train.seed{seed}.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
PY
for seed in 0 1 2; do
  scripts/next.sh python scripts/check_w3_configs.py --runtime train \
    --config "$DATASET_DIR/train.seed${seed}.yaml" || break
done
```

默认 history50、21→7、stateless LSTM128/2、head256/2，CPU 单线程、batch2048、epoch20。当前 CPU torch 不支持 CUDA；若需调 batch/epoch，在 train/val 阶段确定后冻结，512 MB cache 预算不是整进程内存上限。

### 11.3 N3：依次训练三个 seed

```bash
scripts/next.sh python -m factr2_next.training.train --config "$DATASET_DIR/train.seed0.yaml"
scripts/next.sh python -m factr2_next.training.train --config "$DATASET_DIR/train.seed1.yaml"
scripts/next.sh python -m factr2_next.training.train --config "$DATASET_DIR/train.seed2.yaml"
```

每条成功后登记打印的 `saved /绝对路径`，错误时停止检查，不继续假定成功。每个 run 应完整包含 model.pt、config.yaml、normalization.npz、metrics.json、metadata.json。训练按 validation normalized MSE 最小保存 checkpoint；归一化仅 fit train。三种子也按最低 min(val_loss) 选择，平局 seed 小者；在看 test 前记录规则。

## 12. 冻结预算，评估独立 test

```bash
RUN_DIR=/home/dingyj/factr2/runs/SET_SELECTED_SAVED_RUN
scripts/next.sh python -m factr2_next.inference.offline_eval \
  --run-dir "$RUN_DIR" --split val --output "$DATASET_DIR/eval_val"
cp -n config/w3/acceptance.template.yaml "$DATASET_DIR/acceptance.yaml"
```

用工作需求及 pilot/val 无接触分布填写 acceptance 全部字段：version、七关节 RMSE/bias Nm 预算、baseline_improvement、七关节 sigma_floor_nm、free_contact_false_positive_time_rate、filter_alpha、contact_scale_nm、contact_high/contact_low。没有自动通用的正确 Nm 阈值，null 不能 PASS。20% baseline 改善和≤1% 误报是任务书工程起点，七关节绝对预算需实际确定。

在线/离线按 EMA、filtered 的 L1/scale 和迟滞对齐：magnitude≥high 开，≤low 关，中间保持。模板 alpha=.2/scale1/low1/high2 只是占位起点，不是已校准阈值。先 train/val 选滤波/候选阈值，用预声明调参接触试验确认，再冻结；调参接触不进入自由运动训练，也不计正式接触试验。

```bash
sha256sum "$DATASET_DIR/split.json" "$DATASET_DIR/acceptance.yaml" \
  "$DATASET_DIR"/train.seed*.yaml "$DATASET_DIR"/profile/*.yaml > "$DATASET_DIR/frozen.sha256"
date -u +%FT%TZ > "$DATASET_DIR/frozen_at_utc.txt"
scripts/next.sh python -m factr2_next.inference.offline_eval \
  --run-dir "$RUN_DIR" --split test --acceptance "$DATASET_DIR/acceptance.yaml" \
  --output "$DATASET_DIR/eval_test"
```

产物 report.json/metrics.csv/samples.csv/plots；完全配置且满足自动条件退出0，UNCONFIGURED/FAIL 退出2。不带 acceptance 只出指标，不代表通过。看 test 后再调参/补数据，旧 test 已成为开发数据，需要新的独立 test。

自动 PASS 之外还需核对关键姿态/速度/session 分组绝对预算、各动态关节 baseline、近常量关节噪声 floor；现有自动判定不全覆盖。离线误报是有效样本比例，正式在线还要按时间区间统计并报告 invalid 时间。合格模型须 source=real，不能用现有 synthetic smoke 权重代替。

## 13. 合格模型的单侧只读推理

W3 控制端/健康器/real adapter 保持运行，位置模式 active，当前负载/工具/标定/增益/前馈与模型 profile 一致。加载器不自动读取现场硬件配置来验证这些一致性，需人工记录。

### 13.1 N3：由冻结预算生成部署配置

```bash
DATASET_DIR=/home/dingyj/factr2/data/left/real_empty_v1
RUN_DIR=/home/dingyj/factr2/runs/SET_QUALIFIED_SAVED_RUN
ACCEPTANCE="$DATASET_DIR/acceptance.yaml"
DEPLOY_DIR="$DATASET_DIR/deployment"
mkdir -p "$DEPLOY_DIR"
scripts/next.sh python - left "$RUN_DIR" "$ACCEPTANCE" "$DEPLOY_DIR" <<'PY'
import json, sys, yaml
from pathlib import Path
side, run, accepted, dest = sys.argv[1:]
run, dest = Path(run).resolve(), Path(dest)
meta = json.loads((run / 'metadata.json').read_text())
a = yaml.safe_load(Path(accepted).read_text())
assert meta['source'] == 'real' and meta['side'] == side
assert a.get('version') and all(a.get(k) is not None for k in
    ('filter_alpha', 'contact_scale_nm', 'contact_high', 'contact_low'))
c = yaml.safe_load(Path(f'config/w3/{side}/inference.yaml').read_text())
c['checkpoint_dir'] = str(run)
c['smoothing'].update(enabled=True, mode='ema', ema_alpha=a['filter_alpha'], sample_hz=50.0)
c['normalized_contact_magnitude'].update(source='filtered_external_joint_torque', norm='l1', scale=a['contact_scale_nm'])
c['contact'].update(enabled=True, low_threshold=a['contact_low'], high_threshold=a['contact_high'])
(dest / 'inference.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
v = yaml.safe_load(Path(f'config/w3/{side}/visualize.yaml').read_text())
(dest / 'visualize.yaml').write_text(yaml.safe_dump(v, sort_keys=False))
PY
scripts/next.sh python scripts/check_w3_configs.py --runtime inference --config "$DEPLOY_DIR/inference.yaml"
scripts/next.sh python scripts/check_w3_configs.py --runtime visualize --config "$DEPLOY_DIR/visualize.yaml"
```

### 13.2 N4：单侧 inference/web 长期运行

```bash
scripts/next.sh ros2 launch factr2_next readonly.launch.py side:=left \
  inference_config:=/home/dingyj/factr2/data/left/real_empty_v1/deployment/inference.yaml \
  web_config:=/home/dingyj/factr2/data/left/real_empty_v1/deployment/visualize.yaml
```

该 launch 只启动推理和网页，健康器/adapter 需先启动。

```bash
# N3：查看状态与输出
scripts/next.sh ros2 topic echo /next/left/status --qos-reliability best_effort --once
scripts/next.sh ros2 topic echo /next/left/external_joint_torque/raw --qos-reliability best_effort --once
scripts/next.sh ros2 node info /next_inference_left
```

打开 [左臂 NEXT](http://127.0.0.1:8080)。预热连续50帧约1秒，之后应 valid、history_count=50；名字 left_joint_0..6。free/raw/filtered 的 JointState.position 存 Nm。status 的新 header 不代表旧 torque 有效，要看 last_output_stamp_ns/source age。invalid/stale 停有效 torque，恢复重新50帧；contact=false 在无效状态不证明无接触。网页断线可能停留旧曲线。

## 14. 在线无接触与接触验收

### 14.1 N3：新试验目录与身份归档

```bash
LIVE_DIR=/home/dingyj/factr2/reports/generated/8/left_trial_01
mkdir -p "$LIVE_DIR"
cp "$ACCEPTANCE" "$LIVE_DIR/acceptance.yaml"
sha256sum "$RUN_DIR/model.pt" "$RUN_DIR/config.yaml" "$RUN_DIR/normalization.npz" \
  "$RUN_DIR/metrics.json" "$RUN_DIR/metadata.json" \
  "$DEPLOY_DIR/inference.yaml" "$DEPLOY_DIR/visualize.yaml" "$ACCEPTANCE" > "$LIVE_DIR/identity.sha256"
scripts/next.sh ros2 node info /next_inference_left > "$LIVE_DIR/inference_topology.txt"
scripts/next.sh ros2 node info /w3_next_adapter_left > "$LIVE_DIR/adapter_topology.txt"
git rev-parse HEAD > "$LIVE_DIR/factr2_commit.txt"
git -C /home/dingyj/w3_dual_arm_ws rev-parse HEAD > "$LIVE_DIR/w3_commit.txt"
```

沿第7节保存本次实际 W3 参数。

### 14.2 W4：新正式 bag

```bash
LIVE_DIR=/home/dingyj/factr2/reports/generated/8/left_trial_01
ros2 bag record -o "$LIVE_DIR/bag" \
  /w3_robot_bridge_node/state /joint_states /joint_position_controller/command_state \
  /factr2/w3_health /factr2/left/adapter_status \
  /factr2/left/joint_pos /factr2/left/joint_vel /factr2/left/joint_cmd /factr2/left/joint_effort \
  /next/left/free_joint_torque_pred /next/left/external_joint_torque/raw \
  /next/left/external_joint_torque /next/left/mse \
  /next/left/normalized_contact_magnitude /next/left/contact_state /next/left/status
```

每侧先至少10分钟无接触静态/动态观测，覆盖声明姿态、速度、温升，记录全部 valid/invalid 时间。需要 Nm 指标时同时按采集流程录一份新的无接触 H5，正常关闭/strict通过后：

```bash
LIVE_FREE_H5=/home/dingyj/factr2/data/left/real_empty_v1/SET_CLOSED_LIVE_FREE.h5
scripts/next.sh python -m factr2_next.inference.offline_eval \
  --run-dir "$RUN_DIR" --h5-path "$LIVE_FREE_H5" --acceptance "$ACCEPTANCE" \
  --output "$LIVE_DIR/free_eval"
```

进入接触试验前停止自由运动 H5；bag 继续。预先制定允许接触位置/方向、两个姿态、停止条件、独立标记方法与精度。现场确认允许范围后做轻微可重复接触、保持至少0.5s、撤离，每姿态5次，每侧至少10次。不是靠执行一条命令自动完成这一步。

事件表建议存 `$LIVE_DIR/events.csv`：

```csv
trial_id,side,pose_id,direction,contact_start_ros_ns,release_ros_ns,marker_source,video_ref,notes
```

以独立事件/视频标记实际触碰和撤离，不使用 contact_state 自己标真值。Bool/Float32 无 header，需接收时间和新鲜 valid status 关联；全部时间同一时钟，use_sim_time=false。

任务书工程起点：

| 项目 | 现场通过条件 |
|---|---|
| 自由残差 | 七关节及关键分组满足冻结 RMSE/bias 预算 |
| 误报 | valid 无接触时间内 contact=true≤1%；另报事件/invalid 时间 |
| 检出 | ≥9/10 次开始后≤0.5s开启；raw L1超匹配基线P99、filtered magnitude≥high持续≥0.2s |
| 恢复 | ≥9/10 次撤离后≤1s关闭并回匹配基线范围 |
| 速度/延迟 | 有效输出≥45Hz，推理P95<20ms，源→输出接收P95≤60ms |
| 断流 | ≤0.27s stale、停有效torque、恢复重新50帧 |
| 双侧 | 两合格模型同时≥10分钟，命名空间/端口独立 |

这些是待现场验证的要求，不是已达到性能。现有 offline_eval 提供 Nm 及样本比例误报；仓库尚无完整正式事件/时间比例/端到端延迟分析 CLI，需补分析工具或可核验人工/视频分析记录。不能用网页“看起来正常”替代统计验收。

## 15. 右侧与双侧

右侧将数据目录、模板、side、七关节名、话题 root 全部改为 right，独立采集/训练/验收；只换 side 而沿用左权重会被拒收。右侧 NEXT 网页端口8081。

左右单侧均通过后，保持 W3/健康器，另开 NEXT 终端运行右侧 real adapter：

```bash
scripts/next.sh ros2 launch factr2_w3_adapter adapter.launch.py side:=right profile:=real
```

正常结束已有单侧 readonly launch 后，再启动一次 dual launch，避免重复推理节点和端口：

```bash
scripts/next.sh ros2 launch factr2_next dual_readonly.launch.py \
  left_inference_config:=/home/dingyj/factr2/data/left/real_empty_v1/deployment/inference.yaml \
  left_web_config:=/home/dingyj/factr2/data/left/real_empty_v1/deployment/visualize.yaml \
  right_inference_config:=/home/dingyj/factr2/data/right/real_empty_v1/deployment/inference.yaml \
  right_web_config:=/home/dingyj/factr2/data/right/real_empty_v1/deployment/visualize.yaml
```

双侧正式 bag 将第14节所有 `/factr2/left`、`/next/left` 话题的 right 对应项也加入，W3 raw/health仍一份。若做单侧进程隔离测试，用两独立 readonly 终端；dual launch 的 Ctrl-C 会同时停两侧。

观察链路故障测试：仅停止本侧 NEXT adapter，确认本侧推理停止有效输出、W3 控制器模式/目标源不变、另一侧继续；重启 real adapter 后重新50帧。不要为 NEXT 验收拔 CAN、停 bridge、失能电机或在真实domain启动mock发布者。

## 16. 正常收尾与排障

1. 停 NEXT 录制，正常关闭 recorder，等 H5/metadata 写完。
2. 结束 NEXT 推理；停止本次 bag，检查 bag消息数和 H5质量。
3. 现场人员支撑全臂，停止 W3 回放/遥操作，再网页“停止控制端”，检查真实失能；外部控制端由其原终端停止。
4. 最后结束健康器、adapter、UI、bridge。下次同端口启动前确认上个服务已退出。

关闭浏览器不停止运动；停止 NEXT 不停止 W3；停止控制端可能失去重力托举。Ctrl-C、CAN watchdog阻尼都不等于物理急停。硬件异常按现场急停/支撑流程处理，不等待录制或统计结束。

| 现象 | 先检查什么 |
|---|---|
| can0/can1不存在 | 适配器连接、SocketCAN驱动、接口名/物理通道 |
| raw无反馈 | CAN FD参数、供电/ID、使能状态；使能前不一定有完整反馈 |
| q方向/姿态不符 | 当前实际offset/sign路径、装配、标定；先停驱动 |
| 重力下自运动/抖动 | 实物/URDF/夹爪质量、标定、重力摩擦增益；先支撑并停止 |
| command_state无数据 | position是否active、domain、安装的控制器版本 |
| adapter不发四流 | status的reason、两源年龄/skew、health；不要换mock |
| health非OK | 七slot是否齐全、online/enabled/ERR1、raw年龄 |
| recorder不能开录 | 四发布者、fresh frame、metadata路径、同domain |
| metadata没有生成 | 是否Ctrl-C正常close；停止r仅flush |
| 模型拒绝加载 | side、21/7/history50、artifact完整性和哈希、归一化来源 |
| warming不结束 | 连续50帧是否总被gap/健康/adapter异常打断 |
| contact=false但stale | 输出无效，不能判作“没有接触” |
| ABI/包来源不符 | 关闭污染shell，分别用W3系统环境和NEXT next.sh |

快速定位命令（W5）：

```bash
ros2 node list
ros2 control list_controllers
ros2 topic info /joint_states --verbose
ros2 topic info /joint_position_controller/command_state --verbose
ros2 topic echo /factr2/w3_health --qos-reliability best_effort --once
```

NEXT 状态定位（N3）：

```bash
scripts/next.sh ros2 topic echo /factr2/left/adapter_status --qos-reliability best_effort --once
scripts/next.sh ros2 topic echo /next/left/status --qos-reliability best_effort --once
```

## 17. 对照源码与原操作文档

| 依据 | 对应内容 |
|---|---|
| [现有操作总览](README.md) | 环境、终端分工、网页 |
| [第7部分采集训练](07_collection_and_training.md) | profile、录制、质量、split、三seed、预算 |
| [第8部分部署接触](08_readonly_and_contact.md) | 部署生成、LIVE-01..08、事件与时延限制 |
| [W3主手册](/home/dingyj/w3_dual_arm_ws/src/README.md) | CAN、标定、使能、控制/示教 |
| [W3网页手册](/home/dingyj/w3_dual_arm_ws/src/docs/WEB_CONSOLE.md) | 页面步骤与控制边界 |
| [网页管理源码](/home/dingyj/w3_dual_arm_ws/src/ieir_bringup/scripts/web_console_processes.py) | UI启动gravity控制端、读取实际单/双臂offsets |
| [位置控制器](/home/dingyj/w3_dual_arm_ws/src/ieir_controllers/src/joint_position_controller.cpp) | 激活保持、插值、command_state、inactive停发 |
| [硬件接口](/home/dingyj/w3_dual_arm_ws/src/ieir_controllers/src/dm_hardware_interface.cpp) | enable/disable、符号和offset、反馈effort |
| [控制端launch](/home/dingyj/w3_dual_arm_ws/src/ieir_controllers/launch/dual_arm.launch.py) | 增益覆盖、gripper质量处理、控制器加载 |
| [adapter](/home/dingyj/factr2/factr2_w3_adapter/factr2_w3_adapter/adapter_node.py) | 只读四流和诊断 |
| [健康器](/home/dingyj/factr2/factr2_w3_adapter/tools/w3_health_monitor.py) | 系统Python、raw自定义消息 |
| [recorder](/home/dingyj/factr2/factr2_next/src/factr2_next/factr2_next/data_collection/recorder_node.py) | 录制服务、fresh门禁、分段和关闭 |
| [H5 writer](/home/dingyj/factr2/factr2_next/src/factr2_next/factr2_next/data_collection/h5_writer.py) | close后哈希与侧车 |
| [严格质量与manifest](/home/dingyj/factr2/factr2_next/src/factr2_next/factr2_next/data_collection/quality.py) | 元数据、哈希、profile、split隔离 |
| [W3 trainer](/home/dingyj/factr2/factr2_next/src/factr2_next/factr2_next/training/w3_train.py) | 归一化、val选择、五项artifact |
| [推理源码](/home/dingyj/factr2/factr2_next/src/factr2_next/factr2_next/inference/inference_node.py) | 50帧、状态门禁、EMA/迟滞 |
| [离线评估](/home/dingyj/factr2/factr2_next/src/factr2_next/factr2_next/inference/offline_eval.py) | Nm指标、自动验收及其范围 |

最终交付每侧真实数据/profile/split和哈希、三seed结果、合格run五项文件、冻结预算、真实bag/events/video、质量及分组/在线统计报告。首次pilot通过证明采集链路可用，不自动完成模型或双臂真机复现。
