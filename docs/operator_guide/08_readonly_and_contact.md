# 第 8 部分：只读部署、无接触基线与接触验收

前提是 [第 7 部分](07_collection_and_training.md) 已为对应侧给出合格**真实**模型、适用 profile、冻结预算和阈值。先单侧，再双侧；只有两侧都通过才完成第一阶段复现。环境沿用 [总览](README.md)，不额外 source 工作区。

## 1. 部署前核对身份

**W3-检查 / NEXT-工具：**先核对下表。当前加载器会校验模型 artifact 哈希、维度和 side 等，**不会自动读真机当前工具、标定和增益来比较**，这些需要人工核对并保留记录。

| 对象 | 必须相符的内容 | 不相符时 |
|---|---|---|
| W3 实机与第 7 部分 profile | 无夹爪/空载、装配/URDF、本机标定及实际 KP/KD、重力/摩擦前馈 | 停止 NEXT 估计，调查或建立新数据/profile，不换权重名称掩盖 |
| run 的 `metadata.json` | `source: real`、side、joint_order、tool/gripper/load/calibration、dataset 和 manifest SHA256 | 返回模型验收，合成 smoke 不能用于正式结论 |
| 五项 artifact | model/config/normalization/metrics/metadata 全部来自同一 run | 恢复完整 run，不混用不同模型的归一化 |
| 输入 | position active、health/adapter OK、四路七关节、50 Hz、源时间推进 | 按采集页排障，不关闭健康门禁 |
| 接触参数 | 版本、滤波、norm/scale、high/low 与正式测试前冻结文件一致 | 在 train/val/预声明调参阶段确定后重新冻结 |

## 2. 从冻结预算生成部署配置

**NEXT-工具：**以下以左侧为例。把 run 路径设为第 7 部分已经验收的实际 `saved` 路径；右侧用其独立 run、数据集和 `right` 模板重复。`SET_*` 必须替换，不能从目录中自动挑“最新模型”。

```bash
DATASET_DIR=/home/dingyj/factr2/data/left/real_empty_v1
RUN_DIR=/home/dingyj/factr2/runs/SET_QUALIFIED_SAVED_RUN_DIR
ACCEPTANCE="$DATASET_DIR/acceptance.yaml"
DEPLOY_DIR="$DATASET_DIR/deployment"
mkdir -p "$DEPLOY_DIR"
scripts/next.sh python - left "$RUN_DIR" "$ACCEPTANCE" "$DEPLOY_DIR" <<'PY'
import json, sys, yaml
from pathlib import Path
side, run, accepted, dest = sys.argv[1:]
run, dest = Path(run).resolve(), Path(dest)
a = yaml.safe_load(Path(accepted).read_text())
meta = json.loads((run / 'metadata.json').read_text())
assert meta['side'] == side and meta['source'] == 'real'
assert a.get('version') and all(a.get(k) is not None for k in
    ('filter_alpha', 'contact_scale_nm', 'contact_high', 'contact_low'))
c = yaml.safe_load(Path(f'config/w3/{side}/inference.yaml').read_text())
c['checkpoint_dir'] = str(run)
c['smoothing'].update(enabled=True, mode='ema', ema_alpha=a['filter_alpha'], sample_hz=50.0)
c['normalized_contact_magnitude'].update(
    source='filtered_external_joint_torque', norm='l1', scale=a['contact_scale_nm'])
c['contact'].update(enabled=True, low_threshold=a['contact_low'], high_threshold=a['contact_high'])
(dest / 'inference.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
v = yaml.safe_load(Path(f'config/w3/{side}/visualize.yaml').read_text())
(dest / 'visualize.yaml').write_text(yaml.safe_dump(v, sort_keys=False))
PY
scripts/next.sh python scripts/check_w3_configs.py --runtime inference --config "$DEPLOY_DIR/inference.yaml"
scripts/next.sh python scripts/check_w3_configs.py --runtime visualize --config "$DEPLOY_DIR/visualize.yaml"
```

成功判据：真实 side 匹配、预算已冻结、两个配置校验退出 0。模板左/右端口分别 8080/8081；修改端口时分别记录。上面的预算存在检查只是生成入口，完整物理预算还须满足第 7 部分验收。

| 冻结字段 | 推理字段 / 数学含义 |
|---|---|
| `filter_alpha` | `smoothing.ema_alpha`；EMA 首帧取 raw，之后 alpha×raw + (1-alpha)×上次结果 |
| `contact_scale_nm` | `normalized_contact_magnitude.scale`，必须 >0；magnitude = sum(abs(filtered))/scale |
| `contact_high` | `contact.high_threshold`；关闭时 magnitude ≥ high 才开启 |
| `contact_low` | `contact.low_threshold`；开启后 magnitude ≤ low 才关闭，low..high 间保持 |

与当前离线 acceptance 对齐使用 **EMA + filtered + L1**。`scale=1/low=1/high=2/alpha=.2` 等模板值是占位起点，不能直接称为 W3 已标定阈值。其他滤波/norm 虽可运行，若改用它们须重新验证对应阈值与分析方法；不要沿用这份 EMA/L1 验收结论。

阈值确定顺序：先在 pilot/val 决定 alpha 与 scale，按 episode 重置 EMA，再从无接触 filtered magnitude 分布选 high 的高分位候选、比它低的 low。用完整 val 序列按同一迟滞规则验证误报，再用预声明的调参接触试验确认响应/撤离。调参接触试验不进入自由空间训练，也不计入正式十次独立试验。高分位本身不保证 ≤1% 时间误报；满足预算后冻结，正式独立试验期间不再调整。

## 3. 启动单侧，读懂输出

**W3 控制端、健康器、对应 real adapter** 沿用采集页并保持运行。网页确认位置控制 active；处于纯重力模式且 command_state 停止更新时，NEXT 停发是预期门禁行为。

**NEXT-推理-left：**

```bash
scripts/next.sh ros2 launch factr2_next readonly.launch.py side:=left \
  inference_config:=/home/dingyj/factr2/data/left/real_empty_v1/deployment/inference.yaml \
  web_config:=/home/dingyj/factr2/data/left/real_empty_v1/deployment/visualize.yaml
```

**NEXT-工具：**打开 [左臂 NEXT 页面](http://127.0.0.1:8080)，检查七个关节 raw/filtered/free；读取状态与拓扑：

```bash
scripts/next.sh ros2 topic echo /next/left/status --qos-reliability best_effort --once
scripts/next.sh ros2 topic echo /next/left/external_joint_torque/raw --qos-reliability best_effort --once
scripts/next.sh ros2 node info /next_inference_left
```

成功判据：预热后 valid、history_count=50，七关节对应 left_joint_0..6，数据年龄正常；节点只发布 `/next/left` 估计/诊断及 ROS 自身话题，service clients 中没有 W3 运动接口。该 launch 只启动 inference/web，不代替健康器或 adapter。

| 状态 | 含义 / 操作 |
|---|---|
| loading / 启动报错 | 模型尚在加载或未通过校验；看 side、artifact/hash/归一化、配置及端口原因 |
| warming | 连续合法帧尚不足 50；前 49 帧没有有效 torque，约 1 s 后首次输出 |
| valid | 当前输出可用于在声明范围内的分析；仍检查数据年龄和健康 |
| invalid | 坏帧、时间异常、>40 ms gap 或 adapter 非 OK；历史、滤波、contact 重置，按 reason 修复 |
| stale | 全部输入不再刷新，约 .25 s 超时；不持续发布旧 torque，恢复后重新预热 50 帧 |
| 网页 disconnected | HTTP/SSE 断开；曲线可能停留在旧值，不当实时数据使用 |

三路 `free/raw/filtered` 的 JointState **position 字段存 Nm**，名字七维、stamp 为窗口末帧源时间。MSE 单位 Nm²；contact magnitude 是 L1/scale 指示量。Float32/Bool 没有 header，必须结合接收时间及 status 判断；invalid/stale 时 contact=false 不等于“已证实无接触”。网页曲线只保留有限历史，正式实验必须另录 bag。

## 4. 归档并做 10 分钟无接触基线

**NEXT-工具：**为每次现场试验建立新目录，保存冻结身份。采集本次实际 W3 配置快照的方法沿用第 7 部分。

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

成功判据：文件可读且属于本次配置，原始日志可追溯。所有变量在新终端需重新设置。

**W3-审计：**设置同一 `LIVE_DIR`，启动独立正式 bag：

```bash
ros2 bag record -o "$LIVE_DIR/bag" \
  /w3_robot_bridge_node/state /joint_states /joint_position_controller/command_state \
  /factr2/w3_health /factr2/left/adapter_status \
  /factr2/left/joint_pos /factr2/left/joint_vel /factr2/left/joint_cmd /factr2/left/joint_effort \
  /next/left/free_joint_torque_pred /next/left/external_joint_torque/raw \
  /next/left/external_joint_torque /next/left/mse \
  /next/left/normalized_contact_magnitude /next/left/contact_state /next/left/status
```

**网页与现场操作者：**每侧先连续至少 10 分钟无接触静态/动态观测，覆盖第 7 部分声明的姿态、速度与温升条件；记录 valid/invalid 时间及原因，不只观察一个静态姿态。成功条件见第 6 节；期间参数不变。出现持续偏置或关键区域大量 invalid，先排查/补数据，不在正式测试中调阈值。

需要复用现有 Nm 评估时，按第 7 部分同时另录本次**无接触** H5，关闭后 strict 检查，再执行：

```bash
LIVE_FREE_H5=/home/dingyj/factr2/data/left/real_empty_v1/SET_CLOSED_LIVE_FREE_FILE.h5
scripts/next.sh python -m factr2_next.inference.offline_eval \
  --run-dir "$RUN_DIR" --h5-path "$LIVE_FREE_H5" \
  --acceptance "$ACCEPTANCE" --output "$LIVE_DIR/free_eval"
```

成功判据：同一 profile、完整合法 episode、绝对预算满足，且人工补齐关键分组检查。这是新现场验证，不修改原冻结 test manifest。进入接触试验前停止这个自由空间 H5 录制；bag 可以继续记录，并用独立事件表区分时段。

## 5. 接触试验：事件必须独立标记

先写下接触位置/方向、两个姿态、无接触对照、操作者、停止条件和时间标记方法。由操作者在现场确认允许范围后执行轻微可重复接触，保持至少 .5 s，再撤离；每姿态五次，每侧至少十次完整试验。不为试验修改自由空间模型、增益或前馈。

在 `$LIVE_DIR/events.csv` 中记录独立事件，建议表头：

```csv
trial_id,side,pose_id,direction,contact_start_ros_ns,release_ros_ns,marker_source,video_ref,notes
```

**现场标记者：**记录实际触碰开始与撤离，而不是指令/口令时刻。采用同机时钟关联的视频或独立事件标记，预声明标记精度和误差；当前默认使用系统 ROS clock，不要混用模拟时间。人工标记若没有足够精度支撑 .5 s 判据，不能宣称通过检测时延。当前仓库还缺完整的事件标记/分析工具，正式验收前需补齐或提供可核验的人工/video 分析记录。

| 每次试验保存 | 判读 |
|---|---|
| 匹配姿态/速度的无接触对照 | 按预声明 L1 算 raw norm 的 P99；不是把所有姿态混成一条阈值 |
| 接触开始、保持、撤离的原始序列 | measured/free/raw/filtered、magnitude、contact、status、health 与视频/事件表 |
| 检出与恢复时间 | 以独立事件为零点；Bool 无 header，按实际接收时间关联新鲜 valid status |
| 异常/无效区间 | 明确报告，不删除不方便的动作或只统计剩余好片段 |

不使用 NEXT 的 contact_state 自己生成 ground truth，不把接触数据写入 `contact.present: false` 的自由空间训练 session。没有力传感器时报告残差响应和检测可辨识性，不能报告绝对接触力精度、六维 wrench 或安全级检测能力。

## 6. 现场通过条件与统计方法

以下为任务书工程起点，正式测试前冻结；需要替代门槛时在最终测试前记录依据和有限数值。

| 条目 | 必须满足 / 保存的证据 |
|---|---|
| LIVE-01 自由残差 | 七关节及预声明关键分组达到第 7 部分 RMSE/bias 预算；保存全量时序与指标 |
| LIVE-02 误报 | valid 的无接触时间内 contact=true 比例 ≤1%；另报事件数、持续时间、全部 invalid 时间与覆盖 |
| LIVE-03 检出 | 每侧 ≥10 次，其中 ≥9 次在开始后 ≤.5 s 开启；接触 raw L1 norm 超过匹配基线 P99，filtered magnitude ≥high 持续 ≥.2 s |
| LIVE-04 恢复 | ≥9/10 次撤离后 ≤1 s 关闭；回到预声明的匹配无接触基线范围，并报告 raw/filtered 延迟差 |
| LIVE-05 速度/时延 | 预热后 valid 输出 ≥45 Hz；推理 P95 <20 ms、源→输出接收 P95 ≤60 ms；另报 P50/P99/max、资源与测法 |
| LIVE-06 断流 | ≤.27 s stale，停有效 torque；恢复重新 50 帧；不得把重置 false contact 当有效无接触 |
| LIVE-07 隔离 | 停 NEXT/adapter 或错误模型启动不改变既有 W3 模式、参数与目标来源；保存前后拓扑与状态 |
| LIVE-08 双侧 | 两个合格 run 同时运行 ≥10 分钟；命名空间/端口独立、七关节归档、无持续积压或资源失控 |

**统计工具应做的最小工作：**

- 用三个 torque 流的末帧 source stamp 关联有效输出；按同一时钟的接收时间计算 `receive_ns-source_ns`。同机应确认 use_sim_time=false/时间基准一致，并注明 bag 时间戳的含义及记录丢包。
- 误报按**时间长度**统计，分母为预声明范围内 valid 无接触时长；不把重复 status 消息当新帧。推理耗时来自 status 的 infer_ms，按新 `last_output_stamp_ns` 去重、检查缺样本；近 512 样本 P95 不是整场实验分位数。
- 关联独立 events，计算检出/撤离时间和持续超阈值时间；保存逐试验结果，不只报平均值。明确无效区间、未检出、未恢复和标记精度。
- 采样真实 inference PID 的 RSS，报告峰值和随时间增长；记录 CPU/device、输入频率、各分位延迟。现有 mock 探针证明软件能力，不能替代真机本次测量。

现有 offline_eval 已提供 Nm 指标与 EMA/L1 无接触离线误报（按有效样本比例）；尚无覆盖上述全部现场时间比例、事件和时延的正式 CLI。补分析工具后跑相关离线回归，再用真实原始证据计算 LIVE-01..08；网页“看起来正常”不替代结果表。

## 7. 双侧、故障恢复与停机

**NEXT-推理：**两侧单独检查通过后，先 Ctrl-C 结束原单侧 readonly launch（不动 W3/健康/adapter），再启动一次双侧 launch，避免重复节点与端口：

```bash
scripts/next.sh ros2 launch factr2_next dual_readonly.launch.py \
  left_inference_config:=/home/dingyj/factr2/data/left/real_empty_v1/deployment/inference.yaml \
  left_web_config:=/home/dingyj/factr2/data/left/real_empty_v1/deployment/visualize.yaml \
  right_inference_config:=/home/dingyj/factr2/data/right/real_empty_v1/deployment/inference.yaml \
  right_web_config:=/home/dingyj/factr2/data/right/real_empty_v1/deployment/visualize.yaml
```

成功判据：8080 左侧、[8081 右侧](http://127.0.0.1:8081)均 valid，run/side/名字正确，重复无接触及双侧 ≥10 分钟检查。正式双侧 bag 在第 4 节 topic 列表基础上加入右侧全部对应 `/factr2/right`、`/next/right` 信号，保留双方数据。

故障先用独立 mock domain 验证（见 [第 6 部分说明](../inference.md)），**不在真实 domain 启动 mock/replay 测试发布者**。现场只做允许的观察进程故障：

| 操作与终端 | 正常现象 | 不满足时 |
|---|---|---|
| NEXT：在本侧 adapter 原终端 Ctrl-C，保持 W3 控制 | 本侧 inference stale、停有效 torque；W3 模式/目标来源不变，另一侧继续 | 停止本次验收，检查是否错误停止 W3 或共享进程 |
| NEXT：重启该侧 real adapter | 新健康/状态/目标到齐后重新预热 50 帧才 valid | 查 status reason，不复用旧历史 |
| 停本侧 inference（单侧分进程试验或明确的 inference 子 PID） | 该网页 stale、另一侧/W3 继续 | 不用泛化 pkill；dual launch 的 Ctrl-C 会停两侧，不适合证明单侧隔离 |
| 错 checkpoint | 已离线证明校验拒收、无 NEXT 有效输出 | 正式记录证据范围；需要现场复核时只启动单独 NEXT 检查进程，不改活跃部署配置 |

单侧隔离实验可用两个独立 `readonly.launch.py` 终端：Ctrl-C 左侧只结束左 inference/web，右侧/W3 保持。若要验证“左页面仍在线但 stale”，只停止已核对的左 inference 子进程，保留 web；记录 PID 与前后状态，勿停止整个双侧 launch 冒充该实验。

不要为 NEXT 验收拔 CAN、停 bridge、失能电机或制造控制故障；硬件断线/坏反馈行为以 mock 证据明确范围。停止 NEXT 不代表 W3 有失效停车联锁。

**正常收尾：**先停自由录制并关闭 H5，结束 NEXT 后再关闭本次审计 bag并检查消息数；由现场操作者支撑全臂，停止 W3 回放/遥操作、停止控制端确认失能，最后停止健康器/adapter、UI 和 bridge。释放端口后才能启动下一份同端口网页。出现硬件异常按 W3/现场急停流程处理，不等待统计报告。

最终按 [第 8 任务书](../development/8_readonly_deployment_and_contact_validation.md) 交付：左右逐项结论、冻结配置与身份哈希、bag/视频/events、完整统计和限制、只读拓扑、复现/恢复步骤。大文件保留项目忽略目录；协议、清单和精简报告提交 Git。第 7 或第 8 一侧失败，项目仍未完成双臂验收。
