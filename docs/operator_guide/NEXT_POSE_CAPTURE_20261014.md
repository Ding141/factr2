# 直接末端目标采集与 NEXT 训练（2026-10-14）

本版入口是 `scripts/w3_pose_protocol.py`，配方为 `config/w3/motions/right_next_pose_protocol_v3.yaml`。沿用 [论文第 4 节、附录 A.1–A.3](https://arxiv.org/html/2606.12406v2) 和 [官方代码](https://github.com/philiphan0109/factr2_next) 的 NEXT 方法：100 Hz、最近 50 帧 `[q, qdot, q_cmd-q]`、每个窗口重置 LSTM 隐状态；标签为无接触实测电机力矩，外力矩为实测减预测。网络、AdamW、物理 MSE 和早停配置保存在 `config/w3/right/train_next_pose_v3.yaml`。

初版离线检查记录保存在 `docs/acceptance/pose_capture_20261014.json`。随后已执行真机采集和初步训练，最新状态见本文末尾及 `docs/acceptance/pose_trial_20261015.json`；尚未通过独立会话、高速覆盖和接触/卸载验收，不能作为已解决现场问题的证据。

## 指令与平滑运动

采集端对每个联动/末端目标只发布一个 `geometry_msgs/PoseStamped`，话题为 `/factr2/right/pose_goal`。同进程的 `w3_pose_motion.py` 桥接器核对当前目标、坐标系、时间戳和控制器起点，使用计划中已冻结的逆解/冗余姿态，向 `/joint_position_command` 发送一次完整的原生关节目标。每个末端目标仅有两个轨迹节点；不发送插值片段，也不循环刷新轨迹窗口。桥接器不是接受任意未审查位姿的通用 IK 服务。

先采逐关节激励，仍符合论文附录 A.3：一次发送包含 low/high/home 的四节点完整关节轨迹，严格保持其他六关节不动。随后采组合姿态、XYZ 往返和 XY 回路，每个末端目标各发送一次。末端目标之间的实际路径由关节五次曲线决定，XYZ/XY 目标不能被解释为严格直线/圆弧插补。

控制器在内部连续执行五次曲线，限制速度、加速度和 jerk，保持位置/速度/加速度连续。每个目标平滑减速到零，没有人为追加等待；需要研究静止与摩擦返回历史时才保持 10 s。控制循环内部的时间连续化仍然必要。实际 `joint_position_controller/command_state` 始终是训练的 `q_cmd` 来源，不能使用稀疏末端目标或尚未执行的最终关节目标替代。

新采集工具需要新版控制器的 `/joint_position_controller/stop`：监测失败或 Ctrl-C 后取消未开始的目标，从当前 q/v/a 连续制动，保留增益和重力补偿。它不是硬件急停，不能在 ROS 通信失效时保证停车。工具在发送第一步前检查该接口及实际控制器限值；旧控制器缺少接口时拒绝运动。

## 速度、覆盖和奇异点

慢/快档速度上限为 10/20°/s，加速度上限为 40/80°/s²，jerk 上限为 120°/s³；规划留 5% 时间裕量。20°/s 是现有 0.4 rad/s 原生控制限值以内的提速候选，尚未验证真机跟踪。10 月 11 日的旧 20°/s 试采曾因跟踪误差超限中断，所以不能仅凭离线通过认定这一速度可用。跟踪误差仍按 3°停止，失败后不自动重发或降速掩盖覆盖不足。

8 个共享任务姿态在 train/val/test 三个独立会话重新录制，角色种子改变关节方向与联动目标。每个任务姿态包含低/高方向返回、静止、联动及运动后静止。单关节范围与 v2 相同；任务连接、返回和末端运动的全部审查样本要求缩放雅可比条件数 ≤20。逐关节激励和初始接近允许经过奇异区，那里仍能训练/估计关节外力矩。

本机几何离线计划每角色 385 段、约 18.6 分钟，不包括服务和消息交接开销。与论文 10 分钟示例不同，这里额外安排了静止/返回诊断。生成结果必须使用当次实时 URDF、当前静止关节目标和左臂实测姿态重新做碰撞审查；离线示例不是现场许可。

正式训练之前检查实际有效窗口：静止 ≥10%，组合 ≥25%，逐关节 ≤60%，快速动态 ≥15%。每关节正、反方向速度 P95 都须 ≥14°/s，各方向达到该速度至少 100 帧。只有标签叫 `fast`、机械臂却没达到速度的数据会被拒绝。验证每轮另报告静止/联动/逐关节的慢快档物理误差，以及同姿态低/高返回的残差均值差，正式流程仍以独立验证物理 MSE 选择模型。

NEXT 关节外力矩无需雅可比逆解，机械奇异点不应阻止其输出。六维末端 wrench 仍须通过可观测性检查；可以使用既有 `w3_endpoint_wrench.py --mode force_only` 观察三维末端点力，但这明确假设外加力矩为零，不适合测拧转力矩。不能通过放宽奇异门限或把无效末端结果写成零来提高“可用率”。

## 运行顺序

先重建控制仓库并按已有启动手册启动新的控制进程，保持关节位置模式和重力补偿；不要激活另一个笛卡尔控制器。本版复用同一关节控制/摩擦参考链路。

```bash
cd /home/venom/dual_arm_robot
source /opt/ros/humble/setup.bash
source install/local_setup.bash
colcon build --packages-select ieir_controllers --executor sequential
```

从当次控制进程导出完整实时 URDF，将当前静止 `command_state` 的七关节角度填写到 `--initial-deg`；不是全零离线反馈。分别生成三个角色计划。生成过程只写文件。

```bash
cd /home/venom/factr2
scripts/next.sh python scripts/w3_pose_protocol.py \
  --config config/w3/motions/right_next_pose_protocol_v3.yaml \
  --urdf /ABSOLUTE/PATH/current_robot.urdf \
  --initial-deg J1 J2 J3 J4 J5 J6 J7 \
  --role train --output /ABSOLUTE/PATH/train_pose.yaml
```

用 `scripts/w3_collision_check.py` 和当次左臂实际姿态审查各计划，保存与计划同名的 `.collision.json`。标定、末端/负载、增益、重力、摩擦及预热条件在三个会话中保持一致。既有[启动和碰撞审查说明](RIGHT_BASELINE_PROTOCOL_20261011.md)仍适用；运动执行会自动按计划的 `execution_mode` 选择新入口。

终端 C 开始录制，终端 D 才发送该计划。下面 D 的 `--send` 会让机械臂实际运动。

```bash
# C：三个角色分别录制；录制期间按 Enter 停录。
scripts/next.sh python scripts/quick_capture.py \
  --side right --domain 74 --frequency 100 \
  --motion /ABSOLUTE/PATH/train_pose.yaml \
  --bag-profile minimal --bag-compression zstd

# D：环境中的 ROS_DOMAIN_ID 须与 C 一致。
ROS_DOMAIN_ID=74 scripts/next.sh python scripts/w3_coverage_motion.py run \
  --file /ABSOLUTE/PATH/train_pose.yaml --send
```

将真实完成的 train/val/test 会话目录填写到 `config/w3/right/baseline_campaign.example.yaml` 的副本。保留独立会话，不混合 v2 窗口发送和 v3 位姿目标录制。整理器会自动选用新的速度覆盖训练模板。

```bash
scripts/next.sh python scripts/prepare_w3_baseline_campaign.py \
  --sessions /ABSOLUTE/PATH/pose_campaign.yaml \
  --output /home/venom/factr2/data/right/POSE_V3_CAMPAIGN_ID

scripts/next.sh python scripts/train_w3_baseline.py \
  --config data/right/POSE_V3_CAMPAIGN_ID/train.yaml \
  --acceptance config/w3/right/baseline_acceptance.yaml
```

独立验证/测试和真实接触—完全卸载验证全部通过后才能判断改善；卸载数据只用于验证，不能进入无接触训练。现有空载均值、P95、返回差和卸载恢复门限保持不变。尚未完成卸载验证时训练流程返回 1、`deployment_ready=false`，但模型和诊断仍会保存。

理论空载外力为零，测量与预测的有限误差不能保证逐帧严格零。如果静止历史完全相同却因预紧/摩擦/热状态存在不同电机保持力矩，论文输入也未必能唯一解释它；通过多方向返回和独立验收发现该问题，不能使用在线自动扣零把真实持续接触消掉。

## 10 月 14–15 日真机执行和初步模型

右腕 J5–J7 的接触不良由现场修复。20°/s 以及部分 16°/s 快速扫描仍触发 3° 跟踪误差停止；门限保持不变，最终使用 `config/w3/motions/right_next_pose_protocol_v3_fast12.yaml`，慢/快上限为 10/12°/s，加速度均为 40°/s²。该配方是本机完成动作的记录，不代表可外推到高速工况；执行前仍须从实际姿态规划并重新审查碰撞。

会话 `20261014_230723_088450` 完成全部 386 个动作，保存 130,341 行、约 1,303.40 s、100 Hz 数据，采样质量与完整动作执行检查通过。原始 H5、bag、控制参数和碰撞报告保存在本地 `data/quick_capture/right/`，不提交 Git。

目前仅有一个匹配的新协议完整会话。因此 `scripts/prepare_w3_pose_trial.py` 提供**初步试训练**入口：完整姿态 0–5 和关节扫描训练，姿态 6 验证，姿态 7 测试。跨划分边界各裁去 1 s，窗口只在各真实连续片段内构建；归一化只由训练集估计。原始 H5 不改写、不插值补样本，短于 50 帧的片段不形成窗口。短目标的事件时钟与采样时钟允许最多两个 100 Hz 边界采样周期，审计记录该试验例外，正式会话整理器不改变。

```bash
scripts/next.sh python scripts/prepare_w3_pose_trial.py \
  --session /ABSOLUTE/PATH/completed_pose_session \
  --output /ABSOLUTE/PATH/new_trial_dataset
scripts/next.sh python scripts/train_w3_pose_trial.py \
  --config /ABSOLUTE/PATH/new_trial_dataset/train.yaml
```

整理器保存原门限的 `production_coverage_audit.json`。当前数据未通过 ≥14°/s 的双向实测速度覆盖；留出姿态也有超出训练覆盖的静态姿态。试训练配置显式关闭独立会话要求和正式覆盖阻断，标明 `trial`、`deployment_ready=false`，不修改正式训练配置或验收门限。不得把本试验当作独立会话验收，也不能据此部署模型。

试验划分得到 train/val/test 各 83,848 / 8,706 / 12,080 个窗口。沿用论文 NEXT 网络、AdamW、物理 Nm² MSE、梯度裁剪和早停，39 轮后早停，最佳为第 19 轮；耗时约 1,159 s。最佳验证 MSE 为 0.030007 Nm²，保留测试集 MSE 为 0.072903 Nm²。测试中静止 J1 偏差最高约 0.325 Nm，同姿态低/高方向返回的 J1 残差均值差约 0.481 Nm，仍超出原工程门限。模型未自动上线，后续需要匹配的独立会话和真实接触/卸载验证。

权重、归一化、训练日志及派生数据保留本地。版本化结果摘要见 `docs/acceptance/pose_trial_20261015.json`；原始数据和模型文件不随代码推送。

10 月 10–12 日的旧 H5、bag 和模型二进制共 52 个文件已按用户授权删除，释放 1.44 GiB；报告和配置保留，详见 [清理清单](../acceptance/cleanup_20261014.json)。10 月 13–14 日的最新原始数据、候选权重及运行记录保留。
