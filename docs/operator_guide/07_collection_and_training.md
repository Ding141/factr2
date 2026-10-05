# 第 7 部分：采集自由运动数据、训练并验收模型

先完成 [环境与 W3 网页入口](README.md)。本页先做左臂，右臂使用右侧模板、名字及目录重复流程；模型和数据不混用。第一次只在左臂做小范围运动，确认后再扩大覆盖。

## 1. 准备一个可追溯的真实 session

**NEXT-工具：**建立同侧数据集目录和本次 session。下列变量只在当前 shell 生效，后面用到它们的 W3-检查、W3-审计、NEXT-recorder 终端都要重复这两行赋值。

```bash
DATASET_DIR=/home/dingyj/factr2/data/left/real_empty_v1
SESSION_DIR="$DATASET_DIR/left_pilot_01"
mkdir -p "$DATASET_DIR/profile" "$SESSION_DIR/audit"
cp -n config/w3/left/record.yaml "$SESSION_DIR/record.yaml"
# 本流程网页启动的是双臂控制端，实际加载 joint_offsets_dual.yaml
cp -n /home/dingyj/w3_dual_arm_ws/src/ros2_ws_config/joint_offsets_dual.yaml \
  "$DATASET_DIR/profile/calibration.yaml"
cp -n /home/dingyj/w3_dual_arm_ws/src/ros2_ws_config/friction_model.yaml \
  "$DATASET_DIR/profile/friction_model.yaml"
```

成功判据：profile 是**当前控制端实际使用文件**的副本。profile 只在创建数据集时复制一次，此后冻结，勿覆盖；每次 session 用 `sha256sum` 核对实际文件与副本。如果改用单臂控制端，应冻结其实际单臂文件并建立对应新 profile，不能继续声称使用双臂文件。

同侧 train/val/test 必须共享完全相同的 `tool/gripper/load/calibration`。当前 manifest 比较 calibration 的**路径及哈希**；同一文件复制到每个 session 的不同路径也会不匹配。统一引用上述 dataset-level profile。保持位置增益、重力/摩擦前馈、URDF/装配不变，并人工检查实际参数；manifest 不自动检查这些运行配置的一致性。

**W3-检查：**在网页切到“关节位置”后保存本次实际参数，不只抄源码默认值：

```bash
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

成功判据：位置控制 active、gravity active、参数文件非空，零偏与摩擦各自原件/副本哈希相同。重点核对 `kp_gains/kd_gains`、`gravity_gains/friction_gains/friction_model_yaml` 和 URDF。若参数节点不存在，先 `ros2 node list`/`ros2 control list_controllers` 核对名称、模式和 domain，不生成空的“有效配置”记录。

**NEXT-工具：**编辑复制的 `record.yaml`，保留模板的 side、joint_names、四个 topics 和 50 Hz 设置，只替换下面这些项。这里给的是修改项，不能代替完整模板：

```yaml
output_dir: /home/dingyj/factr2/data/left/real_empty_v1/left_pilot_01
session_name: left_pilot_01
metadata:
  source: real
  tool: bare_attachment_v1
  gripper: absent
  load: empty_v1
  calibration:
    path: /home/dingyj/factr2/data/left/real_empty_v1/profile/calibration.yaml
  control:
    gains:
      position_params: /home/dingyj/factr2/data/left/real_empty_v1/left_pilot_01/audit/position_params.yaml
    feedforward:
      gravity_params: /home/dingyj/factr2/data/left/real_empty_v1/left_pilot_01/audit/gravity_params.yaml
      friction_model: /home/dingyj/factr2/data/left/real_empty_v1/profile/friction_model.yaml
  trajectory_id: left_pilot_small_motion_v1
  contact:
    present: false
    label: verified_free_motion
  temperature:
    source: unknown_no_sensor
    condition: cold_start
    warmup_minutes: 0
  audit_paths:
  - /home/dingyj/factr2/data/left/real_empty_v1/left_pilot_01/audit/bag
  health_gate_enabled: true
```

| 填写项 | 现场要求 |
|---|---|
| session/trajectory | 唯一 session 名称；记录这次具体运动和轨迹文件路径/哈希。换运动类别可换新文件/session，便于分组 |
| tool/gripper/load | 上例对应已确认的无夹爪、空载；更换工具/负载要建立新 profile，并检查动力学模型 |
| calibration/control | 指向已核对的实际配置快照；不得保留 `SET_*` 或虚构有效增益 |
| contact | 必须由现场确认自由运动；手扶、手推、碰撞、拖线都不是自由空间训练数据 |
| temperature | 有传感器就注明来源和实测值；无接口保留 unknown，记录冷机/预热时长，勿填写猜测温度 |
| audit_paths | 实际 rosbag/日志位置；提前约定，采集后检查文件确实生成 |

```bash
scripts/next.sh python scripts/check_w3_configs.py --runtime record --config "$SESSION_DIR/record.yaml"
```

成功判据：退出 0。此命令检查接口/路径占位符等配置；metadata、标定文件读取及真实健康声明还会在 recorder 启动时检查，校验通过不代表电机已经健康。

## 2. 按顺序启动观察链路

每条长期命令独占一个终端，使用总览中的相应隔离环境；W3 控制端此时已按既有流程启动。

**W3-健康：**

```bash
/usr/bin/python3.10 /home/dingyj/factr2/factr2_w3_adapter/tools/w3_health_monitor.py
```

**W3-检查：**

```bash
ros2 topic echo /factr2/w3_health --qos-reliability best_effort --once
ros2 topic info /joint_position_controller/command_state --verbose
```

成功判据：对应侧 `factr2/w3/{side}` 为 level 0/healthy，raw 时间与年龄有效；位置控制 active，command_state 有唯一预期来源。失败看 raw 的七个 slot、online/enabled/error_flags 和刷新年龄；正常 ERR 编码为 1，不是凭空认定 0 正常。健康器只能检查 bridge 报告的状态，不能证明所有反馈都未冻结。

**NEXT-adapter-left：**

```bash
scripts/next.sh ros2 launch factr2_w3_adapter adapter.launch.py side:=left profile:=real
```

**NEXT-工具：**

```bash
scripts/next.sh ros2 topic echo /factr2/left/adapter_status --qos-reliability best_effort --once
scripts/next.sh ros2 topic echo /factr2/left/joint_effort --qos-reliability best_effort --once
scripts/next.sh ros2 topic hz /factr2/left/joint_pos --window 100
```

成功判据：adapter OK，七个 left_joint_0..6，effort 为 Nm，源 stamp 推进，输出约 50 Hz。CLI hz 只作在线诊断，最终频率以关闭后的 H5 检查为准。停发时先看 status 原因：位置模式、两源 freshness/skew、原始健康，而不是改成 mock 或关闭门禁。

**W3-审计：**再次设置本次 `DATASET_DIR/SESSION_DIR` 后录原始和适配信号。bag 目录应尚不存在，不复用旧 bag：

```bash
ros2 bag record -o "$SESSION_DIR/audit/bag" \
  /w3_robot_bridge_node/state /joint_states /joint_position_controller/command_state \
  /factr2/w3_health /factr2/left/adapter_status \
  /factr2/left/joint_pos /factr2/left/joint_vel \
  /factr2/left/joint_cmd /factr2/left/joint_effort
```

成功判据：bag 开始记录，结束后 `ros2 bag info "$SESSION_DIR/audit/bag"` 有上述消息及预期时长。若 topic 消息数为 0，检查 domain、发布者和 QoS；不要在 NEXT shell 解码 W3 自定义消息。

**NEXT-recorder-left：**设置同一 `SESSION_DIR` 后启动，节点启动时尚未录制：

```bash
scripts/next.sh ros2 run factr2_next next_record --ros-args \
  -p config_file:="$SESSION_DIR/record.yaml" -r __node:=next_recorder_left
```

成功判据：没有 metadata/文件异常，显示 `Press 'r'`。普通 NEXT YAML 用 `config_file`，adapter 的 ROS 参数 YAML 才用 `--params-file`。更改 metadata 后重启 recorder，不能在同一已打开 H5 中静默换 profile。

## 3. Pilot：先证明数据方向和流程正确

1. **网页：**切关节位置，读取当前；只改左臂一个关节的小范围草稿，采用现场确认的低速/长插值时间，检查路径后确认发送。不要先用回零或大范围双臂回放练习。正常应方向、模型姿态与实物一致，另一侧保持预期状态；不一致先停止实验核对标定/单位。
2. **NEXT-recorder：**保持四流正常时按 `r` 开始，看到 `Writing: ...h5` 和 `Started recording ep_...`。也可由 NEXT-工具调用下方录制服务。开启失败就查 fresh frame/publishers，不反复按键造成状态不明。
3. **网页：**执行现场确认的无接触运动，加入正反方向、加减速和保持；同步记录轨迹、姿态、温升条件。pilot 的幅度和速度由现场可用范围决定，本指导不把 URDF 极限当可直接执行的轨迹。
4. **NEXT-recorder：**运动完成先按 `r` 停录，再 Ctrl-C 关闭 recorder；停止审计 bag。正常显示保存路径，产生 `.h5` 与同名 `.metadata.json`。`r` 停录只 flush，**Ctrl-C 关闭后才完成最终哈希/侧车**；此后才检查和建 manifest。

```bash
# NEXT-工具；仅控制录制，不控制运动
scripts/next.sh ros2 service call /factr2/left/record std_srvs/srv/SetBool '{data: true}'
# 结束本段
scripts/next.sh ros2 service call /factr2/left/record std_srvs/srv/SetBool '{data: false}'
```

服务返回 `success: true` 后再操作运动。每次开录建立 episode；断流/坏帧/时间回退会切连续片段，恢复不会补零或拼接历史。异常或发生人工接触就停录并登记；不能确认哪一段干净时整份文件排除，换新 session。

### 需要重复运动时：示教一次，另行录自由回放

先停止 NEXT 录制，在 W3 网页切重力模式，手拖制作运动 YAML；移开手及其他外力后，切位置模式并确认四流，再开启 NEXT H5 录制，最后确认回放。网页回放默认 `time_scale=0.5`、`ramp_in=5 s`；进入轨迹起点也是运动，须检查路径。W3 示教 YAML 不含完整训练四流，NEXT H5 也不能直接作为运动回放文件。

网页示教可能记录十四关节。若需要明确的左臂七关节轨迹，可在**重力模式、position inactive**时用 W3 工具：

```bash
# W3 终端：只记录手拖轨迹；完成后 q 或 Ctrl-C 保存
ros2 run ieir_controllers teach_replay record \
  --output "$PWD/recordings/left_teach_01.yaml" --rate 50 --max-duration 120 \
  --joints left_joint_0 left_joint_1 left_joint_2 left_joint_3 \
  left_joint_4 left_joint_5 left_joint_6
```

确认无其他回放/遥操作目标源，按上面的自由录制顺序准备后，在 W3 终端执行：

```bash
# 会驱动所列关节；不与网页回放同时启动
ros2 launch ieir_bringup replay.launch.py \
  input:="$PWD/recordings/left_teach_01.yaml" time_scale:=0.5 ramp_in:=5.0 publish_rate:=50
```

回放退出时，只有原本由它激活的位置控制器才被它退回；若位置模式事先已 active，可保持 active。结束后查 `list_controllers`。源码中的 `teach_replay --dry-run` 默认先自动切控制器，后面才判断不发布，**不能当无副作用预检**；当前还缺真正离线的轨迹严格校验/生成工具。

## 4. 检查质量，再扩大覆盖

**NEXT-工具：**把 `H5` 设为 recorder 实际打印的已关闭文件完整路径，不填目录：

```bash
H5=/home/dingyj/factr2/data/left/real_empty_v1/left_pilot_01/SET_SAVED_FILE.h5
scripts/next.sh ros2 run factr2_next next_check_h5 check "$H5" --json "$SESSION_DIR/quality.json"
```

成功判据：退出 0、`accepted: true`；四流 `[N,7]`、相同 int64 源时间、无非有限值/重复/回退，45–55 Hz、最大间隔 ≤40 ms，每段 ≥50 行。先从 audit 逐值核对 q/qdot/q_cmd/tau 的单位、方向、源时间与实际运动，再扩覆盖。

| 常见失败 | 处理 |
|---|---|
| `too_short` / `excluded_episode` | 短段不用于训练；可用 `--episodes ep_0000 ep_0002` 明确检查合法段；不要把短段接在一起 |
| `gap` / 时间回退 / 非有限值 | 查 adapter、健康、主机负载和 audit；保留原始数据，排除坏段，不剪行后拼成连续窗口 |
| `sidecar_hash` / `calibration_hash` | 文件关闭后被改动或实际 profile 已变；恢复可核验原件/重新采集，不手改哈希绕过 |
| 缺 metadata / 无效路径 | 重新核对 session 配置、快照及侧车；缺失记录不能宣称真机数据合法 |

每个正式 session 维护覆盖表（可存 session 的 audit 中）。pilot 只能进入 train/val，不能兼任独立最终 test。

| session / split | trajectory / episode | 姿态区与关节集合 | 正反向与实际速度 | 加减速/保持 | 冷/热条件 | 无接触确认、质量与排除理由 |
|---|---|---|---|---|---|---|
| 待填 | 待填 | 单关节/多关节，各计划工作姿态 | 低/中/计划工作速度，rad/s | 各类有效时长 | 温度来源/预热分钟 | audit 路径、操作者、PASS/排除 |

每侧至少三个不同真实 session 分配 train/val/test，各覆盖预声明运动类别及工作条件。时长由覆盖和 validation 误差收敛决定，不以“录够几分钟”代替验收。UI 手动小范围 pilot 后再安排可重复轨迹；补低覆盖区，工具/负载/标定/控制配置变更时重新划定模型适用范围。

## 5. 固定 split，再训练三种子

**NEXT-工具：**将下面三个变量填成已关闭、已接受且来源独立的真实 H5 路径。多个文件可在对应 `--train/--val/--test` 后继续列出。

```bash
H5_TRAIN=/home/dingyj/factr2/data/left/real_empty_v1/SET_TRAIN_FILE.h5
H5_VAL=/home/dingyj/factr2/data/left/real_empty_v1/SET_VAL_FILE.h5
H5_TEST=/home/dingyj/factr2/data/left/real_empty_v1/SET_TEST_FILE.h5
scripts/next.sh ros2 run factr2_next next_check_h5 manifest \
  --dataset-id left_real_empty_v1 --output "$DATASET_DIR/split.json" \
  --train "$H5_TRAIN" --val "$H5_VAL" --test "$H5_TEST"
scripts/next.sh ros2 run factr2_next next_check_h5 verify-manifest "$DATASET_DIR/split.json"
```

成功判据：三份非空、验证 PASS，`(H5 SHA256, episode)` 无重复。复制文件改名不能隔离来源；内容哈希检查也不自动保证三个真实独立 session，操作者仍须审查覆盖表。

manifest 创建默认选文件内所有未自动排除的 episode。**contact 当前是文件级标签**：发生人工接触就整文件不纳入自由空间 manifest，另采新文件，不保留虚假的 `contact.present: false`。对确认全程无接触、但含通讯异常片段的文件，可审查并修改 manifest 的 episode 列表后重新 verify，保留排除记录。不要修改原始 H5。split 固定后不改文件、标定副本或清单内容；它们的哈希会进入模型身份。

复制正式训练模板，填写绝对 manifest、run 名称；默认架构、H=50、20 epoch、batch=2048、CPU 单线程先保留。内存预算 512 MB 只管数据 cache，不是整个 PyTorch RSS；若需改 batch/epoch，在 train/val 阶段决定并冻结。

```bash
cp config/w3/left/train.yaml "$DATASET_DIR/train.base.yaml"
# 编辑 data.manifest 为本数据集 split.json；保留 side 和模型契约
scripts/next.sh python scripts/check_w3_configs.py --runtime train --config "$DATASET_DIR/train.base.yaml"
scripts/next.sh python - "$DATASET_DIR" <<'PY'
import copy, sys, yaml
from pathlib import Path
p = Path(sys.argv[1])
base = yaml.safe_load((p / 'train.base.yaml').read_text())
for seed in (0, 1, 2):
    c = copy.deepcopy(base)
    c['train']['seed'] = seed
    c['save']['run_name'] = f"next_{c['side']}_{p.name}_seed{seed}"
    (p / f'train.seed{seed}.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
PY
```

依次执行下方命令，seed 改为 0、1、2；每次成功出现 `saved /.../runs/...` 后登记实际路径，非零退出就检查原因。现有验证环境是 CPU 版 torch，请求 CUDA 不可用会失败，不静默回退。

```bash
scripts/next.sh python -m factr2_next.training.train --config "$DATASET_DIR/train.seed0.yaml"
```

| seed | saved 绝对路径 | min(metrics.val_loss) | best_epoch | 时间 / peak_rss_mb |
|---|---|---|---|---|
| 0 / 1 / 2（各一行） | 待填 | 待填 | 待填 | 待填 |

每个 run 自动保存 validation normalized MSE 最低的 epoch，并列取最早。三种子之间同样选 `min(val_loss)` 最低的 run，完全相等取 seed 小者；规则在看最终 test 前记录。run 必须有 `model.pt/config.yaml/normalization.npz/metrics.json/metadata.json`，训练会验证重载一致。归一化只来自 train，test 不参与拟合和 checkpoint 选择。

## 6. 冻结预算，评估独立 test

**NEXT-工具：**先对候选模型用 `--split val` 看分组和残差；填写 acceptance，用工作需求及 pilot/val 噪声确定七关节有限的绝对预算，不照抄合成数据误差。

```bash
cp config/w3/acceptance.template.yaml "$DATASET_DIR/acceptance.yaml"
RUN_DIR=/home/dingyj/factr2/runs/SET_SELECTED_SAVED_RUN_DIR
scripts/next.sh python -m factr2_next.inference.offline_eval \
  --run-dir "$RUN_DIR" --split val --output "$DATASET_DIR/eval_val/$(basename "$RUN_DIR")"
```

| acceptance 字段 | 如何填写与冻结 |
|---|---|
| `version` | 非空版本；另记录冻结时间、依据和文件 SHA256 |
| `per_joint_rmse_nm` / `per_joint_abs_bias_nm` | 按 joint_0..6 的七个有限预算，单位 Nm |
| `baseline_improvement` | 工程起点 0.20，含义为整体 RMSE 至少比训练均值 baseline 降低 20% |
| `sigma_floor_nm` | 七个 pilot 噪声/量化起点；当前只用于诊断 sigma 展示，不改变模型归一化或自动 baseline 判定 |
| `free_contact_false_positive_time_rate` | 工程起点 0.01 |
| `filter_alpha/contact_scale_nm/contact_high/contact_low` | 根据 train/val 无接触分布及预声明调参试验确定；对应 EMA + L1/scale + 迟滞，见下一页 |

冻结完整 split、超参数、种子/选择规则、预算与接触配置后，才评估最终独立 test：

```bash
sha256sum "$DATASET_DIR/split.json" "$DATASET_DIR/acceptance.yaml" \
  "$DATASET_DIR"/train.seed*.yaml "$DATASET_DIR"/profile/*.yaml > "$DATASET_DIR/frozen.sha256"
date -u +%FT%TZ > "$DATASET_DIR/frozen_at_utc.txt"
scripts/next.sh python -m factr2_next.inference.offline_eval \
  --run-dir "$RUN_DIR" --split test --acceptance "$DATASET_DIR/acceptance.yaml" \
  --output "$DATASET_DIR/eval_test"
```

正常产物为 `report.json/metrics.csv/samples.csv/plots`；报告包含每关节 RMSE、signed bias、残差 std、P95/P99，MSE 单位 Nm²；baseline 用相同样本比较。`--acceptance` 全填且全局条件满足时退出 0；null 产生 UNCONFIGURED、失败产生 FAIL，二者退出 2。未传 acceptance 时只出指标，不代表验收合格。

**自动 PASS 还不足以完成第 7 部分。** 另检查并归档：

- 每个预声明关键姿态、速度、session/trajectory 分组满足绝对预算；当前报告 speed/pose 仅按范数 0.5 分桶，不能替代现场定义的姿态区域。
- 每个有效动态关节 RMSE 不差于自己的 baseline；近常量关节按 pilot 冻结的噪声 floor 和绝对预算处理。当前 CLI 自动判定没有这些逐关节 baseline/关键分组规则，需补分析或人工逐项核对。
- 当前离线误报按有效样本比例近似；正式现场的时间比例须按时间戳区间统计，并报告 invalid 时间，方法见下一页。
- 覆盖数量、有效时长、排除段、温升条件和部署边界透明；没有 F/T 真值时此处验收的是**自由力矩预测误差**。

若看完 test 后补数据、调模型或放宽门槛，这份 test 已用于开发，重采新的独立 test。左侧通过不代表右侧通过。

交给第 8 部分的最小交付：每侧合格 run 与五项 artifact、数据/profile/预算的绝对路径和 SHA256、三种子结果、完整覆盖和分组结论、适用/未覆盖条件。按 [第 7 任务书](../development/7_real_data_and_model_validation.md) 编写真机验收报告，二进制/标定快照留在忽略目录，仅提交协议、清单和精简摘要。
