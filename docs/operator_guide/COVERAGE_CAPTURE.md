# 自由运动覆盖采集：逐关节、多关节、慢速与快速

实现依据 [FACTR 2 论文 Appendix A.3](https://arxiv.org/html/2606.12406v1)：逐关节覆盖运动范围，执行末端类似笛卡尔运动的多关节轨迹，重复慢速和快速运动。论文未给出这台机械臂应采用的具体角度范围与速度；本工具的数值是为本机选择的保守初始参数。数据用途是 NEXT 的自由空间电机力矩回归，供外力估计使用。

当前代码只完成离线验证，未启动新一轮真机运动。既有 J6 跟踪问题没有通过本工具解决。现有健康与新鲜度检查依据 ROS 消息及健康状态，不能证明底层电机缓存数据的真实年龄；动态采集仍需核对反馈、目标和电机力矩的时间同步。轨迹生成不包含碰撞检测，需现场核对另一条机械臂、支架、桌面和线缆的净空。采集自由运动数据时保持空载、无接触，避免手扶机械臂。

## 默认方案

配置为 `config/w3/motions/left_coverage_conservative.yaml`，角度均为**绝对控制器角度**，J1～J7 对应 `left_joint_0..6`：

| 关节 | 保守工作范围（度） | 采集中心姿态（度） |
|---|---:|---:|
| J1 | -8 ～ 8 | 0 |
| J2 | 0 ～ 16 | 8 |
| J3 | -8 ～ 8 | 0 |
| J4 | -24 ～ -8 | -16 |
| J5 | -8 ～ 8 | 0 |
| J6 | -6 ～ 6 | 0 |
| J7 | -8 ～ 8 | 0 |

这里的“覆盖运动范围”指配置的工作范围，并非硬件全行程。先以慢速平滑移动到中心姿态，再依次逐关节执行“下界→上界→中心”；每种速度重复两轮，其他关节保持中心目标。随后进行世界坐标 XYZ 方向的末端直线往返（±3 mm），以及 XY 平面圆轨迹（半径 3 mm），同样重复两轮慢速和快速，最后返回录制前的控制器目标。

末端轨迹通过展开 URDF 的正运动学、雅可比和位置逆运动学生成，允许末端姿态随路径变化；不是锁定姿态的六维笛卡尔控制。直线转向处分段减速到零；圆的起止使用平滑时间缩放。规划关节路径采用五次时间缩放、20 ms 密集点，并检查实际控制器线性插值对应的速度和离散加速度。无法在工作范围内完成逆解时，整个计划生成失败，不跳过动作、不自动扩大范围。

慢速各关节速度上限 2°/s、加速度上限 12°/s²；快速为 6°/s、24°/s²。“快速”是本保守协议内的相对速度，不代表已经覆盖高动态工况。路径较短时速度不一定达到上限，以 `coverage.json` 的实际运动与速度为准。从七关节零度起始，默认生成 142 段，计划约 810 秒（13.5 分钟），另有检查开销。实际起始姿态会影响过渡时间。

轨迹采样 50 Hz 与现有 adapter/录制配置一致；论文使用 100 Hz，本次没有修改传感器采样链路。后续若升级到 100 Hz，需要一起验证控制消息、反馈吞吐、同步、训练历史的时间长度。

## 1. D：生成并检查计划

保持现有 A/B 运行，在网页切到关节位置模式并等待机械臂与目标静止。规划只读取状态和模型，不使能、不切控制模式、不发送运动。

新终端 D（以下沿用本机 A/B 的 domain 0、localhost 0；如现有环境不同需同步修改）：

```bash
cd /home/venom/dual_arm_robot
source install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
cd /home/venom/factr2
/usr/bin/python3.10 scripts/w3_coverage_motion.py plan \
  --config config/w3/motions/left_coverage_conservative.yaml \
  --live --output data/motion_plans/left_coverage_01.yaml
/usr/bin/python3.10 scripts/w3_coverage_motion.py run \
  --file data/motion_plans/left_coverage_01.yaml
```

第二条仅离线检查，不连接 ROS。计划文件保存全部轨迹点、配置、起始控制目标、展开 URDF 和模型哈希；旁边生成 `.summary.json`。只读规划不会启动录制。输出文件已存在时拒绝覆盖，下一次使用新的文件名。生成后不要通过网页修改目标；否则执行检查会要求重新生成。

无机器人时也能规划：`plan --urdf <展开的URDF文件> --initial-deg 0 0 0 0 0 0 0 --output <新文件>`。URDF 必须带 `ros2_control` 实际硬件限位；不要把未展开的 xacro 或缺少硬件参数的模型当作运行模型。离线计划只有与运行时模型及起始目标一致才能执行；真机建议使用 `--live`。

## 2. C：开录并冻结同一份计划

新终端 C，等待输出“已开始录制”，再执行 D 的运动命令：

```bash
cd /home/venom/factr2
/usr/bin/python3.10 scripts/quick_capture.py \
  --side left --domain 0 --w3-workspace /home/venom/dual_arm_robot \
  --motion data/motion_plans/left_coverage_01.yaml
```

C 沿用既有 H5、rosbag、健康器和 adapter。它从运行时配置冻结实际标定内容与哈希，继续使用本机已迁移的标定；不修改标定或控制器增益。空载默认 `--load empty_v1 --tool bare_attachment_v1`。配置和实际工具、负载必须一致。

## 3. D：执行这份计划

核对现场路径后，由操作者在 D 输入：

```bash
/usr/bin/python3.10 scripts/w3_coverage_motion.py run \
  --file data/motion_plans/left_coverage_01.yaml --send
```

只有 `run --send` 发送目标。整套轨迹在发送前检查硬件限位交集、3° 裕量、工作范围、连续性、速度与加速度。随后匹配 C 的文件哈希与侧别、确认运行模型没有变化、实测起始角度误差≤1.5°且控制目标与计划起点误差≤0.05°。每段结束保持 1 秒，核对实际到位误差≤1.5°；运行中跟踪误差上限 5°。对 J6 的误差如实保存，不补偿标签、不放宽容差或调增益。

发送采用共享绝对 ROS 起始时间的重叠窗口，约每 150 ms 更新未来 600 ms 的轨迹，窗口保留当前时刻之前的点，避免替换轨迹时重置插值起点。调度卡顿超过 200 ms、健康失效、反馈/控制目标消息超过 250 ms、录制 C 退出或目标偏离计划时，停止后续发送，并将失败保存到 `audit/motion_run.json`。每次录制只允许尝试执行一次；失败后重新开 C，必要时重新生成计划。

**Ctrl-C 或检查失败不是硬件急停。已送出的窗口仍可能继续约 0.8 秒（含起始预留时间）；通讯失效时不能保证机器人停车。C 停录也不直接停止机械臂。异常按现场停车与支撑流程处理。**

## 4. C：停录并核对实测覆盖

D 完成后回 C 按 Enter。除了 `quality.json` 和 `audit/motion_run.json`，C 自动生成 `coverage.json` / `coverage.log`，报告实测角度范围、按快慢速度分别统计的逐关节工作范围覆盖比例、各动作实际速度的 95 分位数、目标跟踪 RMSE，以及多关节段的模型末端移动范围。也可离线重算：

```bash
cd /home/venom/factr2
scripts/next.sh python scripts/w3_coverage_motion.py report \
  --session data/quick_capture/left/本次目录
```

H5 原始数据保持原有 `joint_pos`、`joint_vel`、`joint_cmd`、`measured_joint_torque` 七维结构，兼容 NEXT 输入 `[q, qdot, q_cmd-q]` 和实测电机力矩标签。实测末端范围来自 URDF 与实测关节角推算，不是独立末端测量。执行报告按每段 UTC 时间关联 H5 纳秒时间戳，要求 ROS 时钟为本机系统时间。

质量 PASS、动作 PASS 只表示各自检查通过；需比较工作范围是否实际到达、快慢速度是否实际分开，检查静止时间比例、时间同步与外部接触，另采独立会话划分 train/val/test，再判断训练覆盖。尤其不能因为目标覆盖了 J6 范围，就认定实测 J6 动态已经覆盖。

## 5. 验证后扩大范围

复制配置为新的文件，先改 `name`；调整 `working_bounds_deg`、`home_deg`、`repeats`、`speeds` 或笛卡尔幅度，再生成新的计划。不要直接手改密集轨迹点。需要右臂时一起修改 `side: right` 和 `tool_frame: right_attachment_point`，并核对右臂工作范围与净空。

首次缩短验证可以把 `repeats` 改为 1；它仍包含七关节、全部末端路径和两种速度。从零度起始约 421 秒。进一步扩大范围时，保留硬件裕量；如果 IK 失败，检查中心姿态、路径幅度和工作范围，不应盲目提高速度。

原小范围 pilot 的 `w3_motion_sequence.py` 和动作文件仍可使用，其偏移上限没有被本工具放宽。
