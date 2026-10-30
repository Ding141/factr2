# 有针对性的无接触采集：连续运动、启停摩擦、冷热状态

本次新增采集代码，不启动真机、不改控制器增益、不训练或替换已部署模型。右臂运动，左臂保持生成计划时的实际姿态。只有 `w3_targeted_capture.py --send` 会发送运动；计划生成、碰撞检查、查看计划和离线整理均不发送运动。

入口：

| 文件 | 用途 |
| --- | --- |
| `config/w3/motions/right_targeted_free_v1.yaml` | 姿态、关节范围、速度/加速度/jerk 档位、停留和门限 |
| `scripts/w3_targeted_protocol.py` | 冻结一次实验及模型、双臂起始姿态，支持离线 mesh 检查 |
| `scripts/quick_capture.py` | C 终端：100 Hz 原始数据、完整 rosbag、原始 CAN 和独立停止监视进程 |
| `scripts/w3_targeted_capture.py` | D 终端：执行一次冻结计划，监测实际跟踪，逐档验证 |
| `scripts/prepare_w3_targeted_session.py` | 从已关闭的原始日志生成带标签的完整实验片段 |

## 实验设计

`continuous`：单关节和多关节连续激励，在两个姿态附近运动；一个完整波形只发送一份原生 `JointTrajectory`。内部节点保留非零速度、加速度，由已有控制器执行 C2 五次样条，不按窗口补发、不在每个插值点停车。每个波形的起步、连续段、收尾分别标为 `entry_ramp`、`continuous_excitation`、`exit_ramp`。自然换向时瞬时速度为零不算故意停留；波形边界平滑归零。换姿态、初末静置、速度探测单独标记并从训练派生数据排除。

`hysteresis`：每个关节及七关节组合分别从目标姿态的低侧、高侧返回**相同目标**，到位后各停留 3 或 10 秒。条件运动、返回和静置均保留方向标签；静置有 `intentional_stationary_hold=true`。起止位置、速度、加速度连续，启停时间随速度、加速度档位变化。短距离运动不再一律固定 4 秒。两类实验分别录制、分别输出，禁止把故意启停混入连续激励而不加标签。

`warmup`：最低速度档多关节平滑运动至少 5 分钟，作为热机历史保存，不进入训练数据。

默认档位（°/s、°/s²、°/s³）依次为 `(6,15,60)`、`(10,20,100)`、`(10,40,100)`、`(12,30,120)`、`(12,60,120)`；同速度上限下比较不同加速度。14、16°/s 为显式可选扩展。档位是上限，不能当成实际达到的速度；每段保存原生样条导数界、原生目标速度和实际反馈速度统计，原始 q/v 日志供后续独立核查。

每个档位先完成七关节探测，所有关节探测最大跟踪误差均须 ≤2.5° 才进入实验；进入更高档前，上一档全部已执行片段也须 ≤2.5°。执行时误差 >3°、状态超过 250 ms、硬件健康异常、左臂偏离冻结姿态 >1°、采集心跳或原始 CAN 记录失效会请求原生有界停止。参数超过当前原生控制器限速、限加速度、限 jerk 时拒绝执行，不自动放宽门限或修改增益。探测通过只证明该次探测的实际表现；后续较大激励仍持续检查跟踪。

## 当前磁盘与建议批次

开发验证时剩余空间约 1.2–1.4 GiB，以开录前即时检查为准。C 按 `3 MiB/s × 计划时间 + 512 MiB` 预留空间，不足时拒绝开录，运行中继续检查。默认完整连续实验约 18 分钟、完整启停实验约 40 分钟，当前空间不适合整套录制。下面提供约 80 秒的连续短批次和约 124 秒的启停短批次（起始在参考姿态时；额外接近运动增加时间）。先逐批采集、归档后再扩展；本次没有删除已有数据或模型。

## 真机操作

### 1. 环境与计划

机器人应已由现有流程启动、处于位置模式，健康反馈与 `/joint_position_controller/command_state` 正常。清空右臂工作区域、保持空载和无接触，左臂保持不动。停止其他发送目标的程序，外力显示页面可以保留，但不要在页面上同时发运动。

每个终端都设置同一 domain（当前部署为 74）：

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

以下生成第一批连续实验。`--live` 只读取实际机器人模型和双臂姿态；`--check-collision` 使用该计划冻结的左臂姿态检查 mesh。文件名自行取唯一值，不覆盖已有计划。

```bash
bash scripts/next.sh python scripts/w3_targeted_protocol.py \
  --live --family continuous --role train --thermal unclassified \
  --max-velocity 6 --collect-profiles v06_a15 \
  --single-joints 0 --anchor-indices 0 --coupled-patterns 1 \
  --check-collision --workspace /home/venom/dual_arm_robot \
  --output log/targeted_batch01/continuous.yaml
```

这里 `0` 表示 J1；单关节索引为 0–6。该批次包含 J1 与一组七关节连续激励，仍保留七关节速度探测。下一批改为 `--single-joints 1 2` 等，覆盖 J2–J7；`--anchor-indices 0 1` 扩展姿态。取消这些子集选项则为完整实验。

若只离线检查，改用 `--urdf /绝对路径/robot.urdf --initial-deg ...七个角度... --left-pose-deg ...七个角度...`。历史 URDF 和手填姿态的检查不能替代当次真机计划。mesh 检查是离散模型检查，未建模线缆、支撑物和跟踪偏差仍需现场清空。

查看冻结文件，不发送运动：

```bash
bash scripts/next.sh python scripts/w3_targeted_capture.py \
  --file log/targeted_batch01/continuous.yaml
```

### 2. C 终端开始记录

如果健康监视器和 100 Hz reliable adapter 已由现有部署启动，使用：

```bash
bash scripts/next.sh python scripts/quick_capture.py \
  --side right --domain 74 --frequency 100 \
  --motion log/targeted_batch01/continuous.yaml \
  --bag-profile full --bag-compression zstd \
  --reuse-state-components --raw-can-interface can1
```

复用模式会核对 adapter 的真实数据源、100 Hz、健康门控与可靠 QoS，退出时不关闭被复用节点。若尚未运行这些节点，去掉 `--reuse-state-components`，由 C 启动和关闭本次自有节点。不要同时启动第二个 recorder。新流程要求完整 bag 和实际 100 Hz，拒绝 50 Hz、minimal bag 或 validation-only 配置。

C 显示“已开始无接触录制”后再执行 D。若被动 CAN socket 权限、接口、七个电机反馈或内核时间戳有问题，不进入运动；查看本次目录 `can_record.log`。代码不执行 sudo、不重配 CAN、不发送 CAN 帧。

### 3. D 终端执行一次

```bash
bash scripts/next.sh python scripts/w3_targeted_capture.py \
  --file log/targeted_batch01/continuous.yaml --send
```

D 检查模型、增益、原生运动上限、当前双臂姿态、匹配的碰撞报告和 C 会话。连续激励发送完整原生轨迹；两节点的到位运动发一次末端位姿，由现有冻结分支桥转为已检查的原生轨迹。不会按小段反复发送位置。

C 另启独立 guardian，D 必须收到 armed 应答才允许发送轨迹。D 心跳丢失超过 250 ms，或 C 停止更新/退出时，guardian 请求 `/joint_position_controller/stop`。D 自身异常也请求相同停止。此为软件监视，不能替代硬件急停。

D 完成后回到 C 按 Enter，正常关闭 H5、CAN 和 bag。每个 C 会话只允许 D 尝试一次；失败后保留原始日志，排查原因，再从当前实际姿态生成新计划和新会话。

运行中可 Ctrl-C 中止 D，或 Ctrl-C/Enter 结束 C。也可在 C 显示的**本次会话目录**创建 `STOP_MOTION` 文件请求停止。如果发生接触，在同一目录创建 `CONTACT_DETECTED`：会停止运动，并永久阻止该会话作为无接触数据导出。

### 4. 启停实验独立开录

连续实验关闭后，重新生成启停计划并开一个新的 C 会话：

```bash
bash scripts/next.sh python scripts/w3_targeted_protocol.py \
  --live --family hysteresis --role train --thermal unclassified \
  --max-velocity 6 --collect-profiles v06_a15 \
  --single-joints 0 --anchor-indices 0 --coupled-patterns 1 \
  --check-collision --output log/targeted_batch02/hysteresis.yaml
```

C、D 命令换成这份文件，其余同上。同姿态的正反方向返回及停留为配对观测，不能只比较目标角度；还应核对派生数据中实际位置均值、原生目标与力矩。

### 5. 提速、独立验证与冷热机

低速通过后，新批次可选 `--max-velocity 10 --collect-profiles v10_a20 v10_a40`；保留前面档位的七关节探测，比较同速度上限下的不同加速度。继续扩展到 12 时选择 `v12_a30 v12_a60`。14、16 仅显式选择后才纳入，并保留前序探测和门限。先前曾出现较高速度跟踪超限，不应直接跳到最高速度。任意一档超限即停止并保存证据。

训练数据用 `--role train`；另行采集 `--role val` 和 `--role test`，使用独立 C 会话。连续运动按 role 使用不同随机相位；启停实验保持同一目标对照。不会把同一次录制按行随机拆成训练/验证。延迟校准应沿用验证顺序 1 的训练数据冻结结果；本采集器不重新拟合或应用时间平移，不修改已有延迟设置。

冷热条件必须如实声明：

* `--thermal unclassified`：未完成温度条件设计的常规调试。
* `--thermal cold_start --idle-minutes 30`：操作者确认至少静置 30 分钟。它表示“冷启动条件声明”，不证明电机已达到室温；刚完成上一批运动不能再次标为 cold_start。
* `--family warmup --thermal unclassified --max-velocity 6 --warmup-minutes 5`：另开 C/D 记录至少 5 分钟低速热机。随后新计划用 `--thermal warm_start --warmup-session data/quick_capture/right/实际热机会话目录`。要求同一模型的热机完整成功，且完成至执行不超过 5 分钟，否则拒绝。当前剩余空间不足以安全预留完整热机，须先准备存储空间。

两种实验都分别采集冷启动和热启动批次；真正的冷启动批次需重新静置。温度在录制中可能上升，cold/warm 标签只描述初始条件。原始 CAN 被动记录 D6/D7 名义 MOS/转子温度及有效性，派生数据逐样本保留温度和缺失掩码；它不是经校准的关节温度探头，也不自动证明达到热稳态。

### 6. 关闭后生成带标签数据

把 C 输出的实际会话目录填入：

```bash
bash scripts/next.sh python scripts/prepare_w3_targeted_session.py \
  --session data/quick_capture/right/实际会话目录 \
  --output data/right/targeted/唯一输出目录
```

输出 `samples.h5`、metadata、`quality.json`、`source_ranges.json`。保留 FACTR2 NEXT 四个原始流：q/c 为 rad，v 为 rad/s，力矩为协议名义 N·m；额外标签位于每个 episode 的 `labels` 组。训练输入仍为原始四流，温度/相位不是自动添加的网络输入。

整理器要求完整成功的真实 100 Hz 无接触日志、正确的冻结计划/碰撞报告、干净关闭且无接收丢帧的原始 CAN；严格按命令的 ROS 时间区间选原始行，不填样本、不插补力矩、不混合实验。探测、转场和热机从派生训练数据排除。启停和连续数据分别输出并保留各自 role。

如训练会话中途停止，`--completed-only` 只取已经完整完成的实验块；val/test 禁止该选项。日志间断或实验片段缺失时拒绝导出。导出失败不修改原始日志，修复原因后使用新的输出目录，不能覆盖不完整的派生目录。

MD 等长历史特征应先在完整原始会话上因果计算，再按 `source_ranges.json` 选片段；遇原始采样间断或独立会话要重置，不能把单个切片当成没有前史。Linux 内核 CAN 接收时间、用户接收时间和 ROS header 一并保存，但仍不等于电机内部采样时间。本次不宣称已解决传感器真实时延、温漂或外力误报，新增实验用于区分并验证这些原因。

## 开发验证范围

自动测试覆盖样条 C2 连续和原生多项式约束、实际不同的启停档位、配对停留标签、冻结文件篡改拒绝、提速门限、心跳失效停止、被动 CAN 解码及温度缺失、完整块派生、C 的复用与正常关闭。测试使用假 ROS/进程和人工日志，没有打开 CAN socket 或驱动电机。

离线完整路径检查使用已保存的真实 URDF、右臂参考姿态和左臂全零这一明确假设。结果仅覆盖该冻结模型和姿态，真机前必须重新 `--live --check-collision`。完整验证记录见 `docs/acceptance/targeted_free_capture_20261016.json`。
