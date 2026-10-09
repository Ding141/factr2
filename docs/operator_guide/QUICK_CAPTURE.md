# 四终端采集：D 读取一个动作文件，顺序执行整套动作

沿用你已经跑通的连接方式。A 是 bridge，B 是网页，C 开录/停录，D 执行动作文件。**不用在 D 一条条发坐标。** 脚本直接从源码运行，无须重新编译；本次开发只做了离线检查，没有启动真机控制或发送目标。

本文示例：无夹爪、空载、右臂。动作文件是 [right_all_joints_pilot.yaml](../../config/w3/motions/right_all_joints_pilot.yaml)。它让 **J1～J7 依次 +8°、返回起始目标**，14 段，每段运动 8 秒、保持 2 秒，前后静置共 5 秒，总计划约 **145 秒**，另有服务检查耗时。文件明确设置 `max_step_deg: 10`，支持本次 5～10° 范围的动作；示例取 8°，为到位误差留出余量。

所有目标相对于 D 开始执行时**同一次读取的实测姿态**，不会逐步累加误差。“返回”表示返回这个起始目标角度，实际误差仍取决于控制器。示例只约束右臂；左臂继续由现有控制模式管理，现场核对双臂及线缆的运动净空，尤其是比原来 2° 更大的扫掠范围。软件不做碰撞规划。

**当前验收限制：**格式检查和 D 的新鲜度检查依据消息时间戳，尚不能识别底层硬件缓存中反馈数值的实际年龄。已发现的反馈积压、J6 跟踪不足及停机失能确认超时仍待处理；正式动态采集需先核对原始电机反馈与控制目标的时间同步。两项 PASS 不代表这些问题已修复或训练数据已通过真机验收。

## 1. A/B：保持现有连接

**已经启动就不用重复执行。** 如果尚未启动，A：

```bash
cd /home/dingyj/w3_dual_arm_ws
source install/setup.bash
ros2 launch ieir_bringup bridge.launch.py arms:=dual gripper:=false
```

B：

```bash
cd /home/dingyj/w3_dual_arm_ws
source install/setup.bash
ros2 launch ieir_bringup ui.launch.py workspace:="$PWD" gripper:=false
```

打开 [W3 操作网页](http://127.0.0.1:8766)。由你现场支撑、解锁、启动控制端，再切到 **“关节位置”**。后续采集脚本不使能、不切模式、不回零。执行整套动作期间不要用网页发送目标，也不要同时回放或遥操作。

下面按你原来普通启动的 **ROS domain 0 / localhost 0** 写。若 A/B 实际用了其他值，C/D 跟随已有环境，不为采集重启控制端；例如 domain 73 时，将下面 C 的 `--domain 0` 和 D 的 `ROS_DOMAIN_ID=0` 都改成 73。原 A/B 若 localhost=1，C 运行前 `export ROS_LOCALHOST_ONLY=1`，D 也改成 1；自定义 RMW/DDS XML 同样保持一致。

## 2. C：开录，指定本次动作文件

新终端 C，无须 source 工作区：

```bash
cd /home/dingyj/factr2
/usr/bin/python3.10 scripts/quick_capture.py \
  --side right --domain 0 \
  --motion config/w3/motions/right_all_joints_pilot.yaml
```

等到输出 **“已开始录制”**，再去 D 执行动作。C 保持运行。

C 自动检查现有位置/重力控制器，保存实际参数、标定和动作文件，启动健康器、real adapter、审计 bag 和 H5 录制。它冻结动作文件的内容与哈希；D 使用另一份或改动后的文件会被拒绝。默认记录空载 `empty_v1`、无外部接触；有真实负载时增加 `--load 真实负载标识`，它只记录身份，不校验重力模型。采自由运动时不要手扶、推压或碰撞。

## 3. D：一次命令执行整套动作

新终端 D，先设置与现有 W3 相同的环境：

```bash
cd /home/dingyj/w3_dual_arm_ws
source install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
cd /home/dingyj/factr2
```

首次先检查文件；这条命令完全离线，不连接 ROS、不运动：

```bash
/usr/bin/python3.10 scripts/w3_motion_sequence.py \
  --file config/w3/motions/right_all_joints_pilot.yaml
```

确认现场允许右臂七个关节的 +8° 方向及返回路径后，**由你手动执行真正运动的命令**：

```bash
/usr/bin/python3.10 scripts/w3_motion_sequence.py \
  --file config/w3/motions/right_all_joints_pilot.yaml --send
```

只有加 `--send` 才会发送目标。D 自动读取起始七关节姿态，预检查**全部**目标的实际硬件限位，随后逐段发送、等待、核对到位误差，再执行下一段。输出如：

```text
[1/14] J1_plus8：运动 8s，保持 2s
[2/14] J1_return：运动 8s，保持 2s
...
[14/14] J7_return：运动 8s，保持 2s
整套动作完成；执行报告：…/audit/motion_run.json
请回到 C 按 Enter 停录并检查。
```

每段的 t=0 使用位置控制器最新实际目标，避免把已有跟踪误差当成下一段起点的跳变。本文件每关节目标距当前实测/上一目标均须 ≤10°；运动 3～60 秒。运动过程中，实测与控制器当前插值目标的误差上限仍为 5°，最终到位容差仍为 1.5°。反馈、command_state、real adapter 超过 250ms 或健康失效、C 退出、目标未被控制器保持、误差过大时，停止发送后续段。没有自动重试、自动回起点或调增益。

示例到位容差为 **1.5°**：之前左臂 J7 测试出现约 1° 的稳态误差。右臂本次也先采用这一 pilot 继续条件，实际误差由本次测试核对，并不代表右臂精确定位已经验收。D 保存每一步的真实误差，后续按实测结果调整协议。

每次 C 开录只允许这份动作执行一次。下一轮重新启动 C，再启动 D；旧动作已经尝试过、哪怕失败，也不会直接重复发送。

## 4. C：按 Enter 停录并检查

看到 D 的 **“整套动作完成”** 后，回 C 按 Enter。C 正常关闭录制与 bag，保存 H5 哈希和元数据，检查四流质量，并核对动作执行报告。

正常应有：

```text
H5：/home/dingyj/factr2/data/quick_capture/right/…/….h5
质量检查：PASS
Δq(deg)= […]   Δq_cmd(deg)= […]
整套动作执行：PASS
```

检查结果这样看：

- `质量检查：PASS`：四流均为七维、有限，时间戳一致且递增，45～55Hz，最大间隔 ≤40ms，有效段 ≥50 行，元数据/哈希正确。
- `整套动作执行：PASS`：这份文件的 14 段都已完成并通过 D 的到位核对。中断后，即使部分 H5 质量合格，也会显示整套动作 FAIL。
- `Δq` 是每个关节实测的最大角度减最小角度；`Δq_cmd` 是控制目标范围。这个示例七个目标范围应约为 8°，实测七个关节也应有运动。两种 PASS 加上现场运动/无接触核对，才算本次链路测试完成；它不表示训练覆盖已足够。

同一 session 目录保存 H5、`quality.json`、`audit/motion.yaml`（本次文件快照）、`audit/motion_run.json`（起始姿态、全部解析目标、逐步误差及完成/失败状态）和 `audit/bag/`。

**D 的 Ctrl-C 只停止后续发送，已发轨迹仍可能执行；C 停录也不停止机械臂。** 实物异常按现场停车/支撑流程处理，再停录保留证据。

## 5. 后续只改动作文件

复制示例成新的 YAML，改名字和 `steps`，然后把 C/D 的文件路径一起换成新文件即可，无须改 Python 或编译。开录后不要再修改文件。

一行示例：

```yaml
- {name: J3_minus8, offset_deg: [0, 0, -8, 0, 0, 0, 0], duration_seconds: 8, hold_seconds: 2}
```

含义：**相对本次固定起始姿态，右臂 J3 目标 -8°，其他六关节目标为各自起始角度，8秒运动、2秒保持**。`offset_deg` 的七个数顺序为网页 J1～J7，即 `right_joint_0..6`；全零行是返回起始目标，绝不是机器人回零。每行 `name` 唯一。偏移绝对值和相邻两行每关节差值均不得超过文件的 `max_step_deg`（本文件为 10°，未写时默认 5°，最大可设 10°），保持至少 0.5秒；还要通过真机实际限位及现场路径核对。+8° 到 -8° 相差 16°，需经起始目标 0° 过渡，不能直接相接。

原左臂 2° 文件仍保留。需要测左臂时，将 C 的 `--side` 改为 `left`，C/D 文件路径一起改为 `config/w3/motions/left_all_joints_pilot.yaml`。首次按单侧分别验证。若有 C 已经开录，请先停录，再使用右臂新文件重新开录。

## 6. 没反应时

| 提示/现象 | 下一步 |
|---|---|
| C 没显示已开始录制 | 看 C 打印目录的 `health.log` / `adapter.log` / `recorder.log`；核对关节位置模式及 domain |
| C 尚未用这份文件开录 | C 命令须有 `--motion`，等待已开始录制；C/D 文件内容、侧和 domain 须相同 |
| 本次动作已经运行或尝试过 | 回 C 停录保存结果，下一轮重新开 C，再运行 D |
| 到位误差超限/状态失效 | 不继续盲发；看 `audit/motion_run.json` 中第几段、实际误差及原因 |
| 文件检查 PASS 但不运动 | 检查文件命令未加 `--send` 时本来就只做离线检查 |

已有健康器/本侧 adapter/recorder 时 C 会拒绝重复启动。先结束旧采集进程，不重复启动 bridge 或控制端。
