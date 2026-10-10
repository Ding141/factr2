# 右臂连续自由运动采集与 NEXT 训练

本流程对应 FACTR 2 附录 A.3：先逐关节，再执行多关节笛卡尔类轨迹，重复慢速与快速运动。必须空载、没有外部接触。数值范围是本机扩大后的工作范围，仍未覆盖全部机械行程；不是论文给出的 Piper 角度参数。

配置：[right_coverage_smooth.yaml](../../config/w3/motions/right_coverage_smooth.yaml)。J1/J3/J5/J7 为 ±16°，J2 为 0～32°，J4 为 −40～−8°，J6 为 ±12°。采集中心为 `[0,16,0,-24,0,0,0]` 度。慢/快速度上限 3/10°每秒，加速度上限 12/35°每二次方秒；离散插值对应的 jerk 上限 120°每三次方秒。末端 XYZ 正弦往返 ±8 mm、XY 圆半径 8 mm，允许姿态随位置变化。

旧程序把上下界和回中心拆成独立的停止段，每段再保持 1 秒。新参考曲线采用正弦往返、五次时间缩放和密集样点；只有必要的换向和逻辑块交接处速度自然降到零，没有端点停留。全程使用一个 ROS 起始时刻和连续重叠窗口，逻辑块的记录与到位检查不重新发布起点或插入等待。该阶段的旧控制器仍进行线性位置插值，因此此处 C2 连续性仅指参考曲线；下方 2026-10-10 C2 更新已改为五次 Hermite 控制插值。实测平滑性须通过录制反馈确认，不能仅凭规划曲线认定机械臂消除了摩擦导致的顿挫。

计划约 23 分钟（含起始姿态过渡，时长随姿态变化），重复两轮慢/快，62 个逻辑块；回到录制前目标。轨迹不含碰撞检测。需要现场核对右臂、支架、另一条机械臂和线缆净空。本次左臂保持失能，不发送左臂目标。

2026-10-10 调试中，右臂 J6 原增益 `kp=10、kd=0.6` 在慢速运动时超过 5° 跟踪保护，自动停录。现场确认无干涉后，保持重力补偿运行，仅重新加载位置控制器试用 J6 增益；`20/0.85` 的 ±4° 小幅测试回位误差为 1.75°，未通过。`35/1.0` 的同范围慢/快测试通过，最大跟踪误差约 2.03°、各段到位误差小于 1°、实测跨度 6.12°（指令 8°），峰值实测力矩约 0.82 N·m。该增益已保存到 dual_arm_robot 的 slave 及位置/笛卡尔控制配置，并备份原文件；标定文件没有改变。它只在本机空载测试，不构成其他负载/姿态的增益验证。完整采集仍使用 5° 跟踪保护及 1.5° 到位保护。

因 domain 0 曾出现另一台机器人控制器，本次使用 `ROS_DOMAIN_ID=74`、`ROS_LOCALHOST_ONLY=1`。桥接、控制器、网页、录制和运动必须一致；不能把默认 domain 0 的反馈混入本次采集。

在右臂关节位置控制准备好后：

```bash
cd /home/venom/dual_arm_robot
source /opt/ros/humble/setup.bash
source install/local_setup.bash
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1
cd /home/venom/factr2
/usr/bin/python3.10 scripts/w3_coverage_motion.py plan \
  --config config/w3/motions/right_coverage_smooth.yaml --live \
  --output data/motion_plans/新的右臂计划.yaml
```

C（仅录制）：

```bash
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1
cd /home/venom/factr2
/usr/bin/python3.10 scripts/quick_capture.py --side right --domain 74 \
  --frequency 100 --w3-workspace /home/venom/dual_arm_robot \
  --motion data/motion_plans/新的右臂计划.yaml
```

D 在同一 ROS 环境，确认 C 已开始录制后：

```bash
/usr/bin/python3.10 scripts/w3_coverage_motion.py run \
  --file data/motion_plans/新的右臂计划.yaml --send
```

完成后 C 按 Enter 正常停录。失败不自动重试，不放宽跟踪/到位容差。停止后续窗口不是硬件急停；已经发布的窗口可能继续约 0.8 秒。录制必须正常完成、质量通过后才训练。

采集频率 100 Hz、历史 50 帧，保持原始七维数据，不插值制造新测量。质量检查核验实际频率与间隔；禁止 50/100 Hz 数据混用。训练使用两层 hidden=128 LSTM、两层 hidden=256 MLP、dropout=0.1、AdamW lr=0.001、weight decay=1e-6、梯度裁剪 1.0、验证早停 patience=20/warmup=10/min_delta=1e-5。最多 80 epochs，按最低验证损失保存，CPU 硬件与论文 RTX 3090 不同。

```bash
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1
scripts/next.sh python scripts/prepare_w3_session_trial.py \
  --session data/quick_capture/right/本次目录 --output data/right/新的试训目录
scripts/next.sh python scripts/run_w3_session_trial.py --dataset data/right/新的试训目录
```

第一轮重复用于训练，第二轮完整关节/路径块用于验证和测试，慢/快分配互补，每个连续块两端剔除 1 秒。只有纯过渡段在剔除后不足 50 帧时会跳过并记入审计；正式激励段不足历史窗口则拒绝训练。CPU 线程数为 `min(8, CPU逻辑线程数)`，保存实际配置；本机实测 8 线程的训练批次约比 1 线程快 3 倍。各划分不共享原始帧，但仍来自同一会话，因此只能作为试训；需要独立会话评估及接触参考后才能验证泛化和真实接触力精度。右臂模型不与左臂模型混用，100 Hz 模型不能按 50 Hz 运行。

论文：[FACTR 2 附录 A.1/A.3](https://arxiv.org/html/2606.12406v1#A3)。下方 C2 流程在训练评估后可启动只读在线估计与末端图表；估计不接入机器人控制闭环。


2026-10-10 的 C2 大范围采集补充：位置控制器从轨迹交接状态正确开始，未来开始时间之前保留原轨迹；使用五次 Hermite 插值并衔接位置、速度、加速度，显式限制速度、加速度和 jerk，窗口中断时连续制动。控制周期的 command_state 和原始 JointTrajectory 均录入 rosbag。当前使用 0.6 秒窗口、0.15 秒发送间隔、100 Hz 路径样本；同一关节两次重复连续通过中心点，关节切换与整段结束平滑停下。

新增摩擦前馈使用新鲜的规划速度，在零速附近连续过渡；失效后按变化率限制归零。当前力矩上限为 0.8 N·m/关节、变化率为 8 N·m/s。右臂 J6 采用保守试验系数 0.45 N·m（乘现有 0.8 增益）；该系数是待进一步验证的工作参数，不等同于完成摩擦标定。位置增益、关节方向及零偏未更改。20261010_160004_957524 小范围复测中，J6 慢/快段停滞比例为约 5.6%/1.4%，范围覆盖为 82.0%/83.1%；误差仍存在，不应宣称已消除所有机械黏滑。

正式采集计划 right_coverage_c2_expanded_friction_20261010.yaml 已经用户批准，关节范围约为机械跨度的 80%，慢/快上限 6/20°每秒，多关节末端轨迹 20 mm；预计运动约 36 分钟。重新接入后的初始保持姿态已用于轨迹衔接；标定和 URDF 哈希不变。结束后保持 [0,50,0,-55,0,0,0]°。

正式流程增加离线控制目标连续性检查，并检查每个单关节动作的实际跨度覆盖至少 90%。未通过时停止自动训练并保留原始数据。模型训练快照保存实际控制器源码、摩擦参数、采集诊断及动作事件时间；会话内评价仍不能替代独立会话或已知外力参考测试。

2026-10-10 连续边界检查修复与分段恢复：非零速度的重复中心点使用当前控制目标进行 5° 跟踪检查，不再误用静止终点 1.5° 到位检查；实际停止终点保持原有到位保护。初次大范围录制 `20261010_160331_906917` 在此错误检查处中断，仅其完整完成且通过连续性审计的 J1～J6 慢速段用于训练。补采 `20261010_164151_566506` 完成剩余 J7 慢速、全部快速单关节和多关节慢/快路径，84810 帧，100 Hz，质量与控制目标连续性均通过。

合并训练使用 `prepare_w3_session_trial.py --completed-prefix` 的严格恢复检查，以及 `combine_w3_session_trials.py` 的标定、控制器参数和源文件哈希一致性检查。未完成事件排除；H5 不拼接，历史窗口不跨录制断点，各划分不共享原始帧。本次仍为同一采集活动的重复动作留出，不是独立会话泛化评估。过程、来源和最终模型状态记录于 `log/manual_start/20261010-c2-resume/CAMPAIGN_REPORT.md` 与 `workflow_state.json`。
