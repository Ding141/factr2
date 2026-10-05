# W3 / NEXT 统一契约 w3_next_v1

机器定义：`config/w3/contract.yaml`。所有运行配置、数据清单、模型 metadata 引用此版本。单臂 7 关节，顺序 `{side}_joint_0..6`，left/right 独立采集、训练和推理，夹爪不进入模型。

| key | 输入 → adapter 输出 | JointState 字段 | 单位 |
|---|---|---|---|
| joint_pos | /joint_states → /factr2/{side}/joint_pos | position | rad |
| joint_vel | /joint_states → /factr2/{side}/joint_vel | velocity | rad/s |
| joint_cmd | /joint_position_controller/command_state → /factr2/{side}/joint_cmd | position | rad |
| measured_joint_torque | /joint_states → /factr2/{side}/joint_effort | effort | Nm |

所有消息为 sensor_msgs/JointState。两输入分别按名字重排；各所需数组与输入 name 等长，拒绝重复名/缺少关节/非有限所用值。允许另一侧和夹爪。输出指定字段长度 7，其他字段为空；控制目标输入 position/velocity 等长，effort 为空。目标是本 update 已写入 command interfaces 的插值位置/速度，stamp 为 update(time, period) 的 time，不能等价于真实 CAN 接收时刻。

adapter 50 Hz 选择最近合法两源，两源 ROS stamp 年龄与 monotonic 接收年龄分别 ≤0.25 s，源 skew ≤0.03 s；未来 stamp 容差为 0（同 ROS 时钟）。四流共享选中的 joint_states stamp，每个状态 stamp 最多输出一次。零、重复、回退、未来 stamp 拒收，ROS 时钟回退清缓存和 stamp 基线。无效新消息清空两源缓存，必须两路重新合法才能恢复；静态值、推进的 stamp 合法。断流不补零、不外推。硬件健康恢复同样等待新合法输入。

四流 H5 每行共用 int64 纳秒时间戳，接受片段严格递增，窗口不跨 episode、不跨 >40 ms 的缺口；需切段或拒收，不能删坏行拼接。后续 recorder/dataset 工具负责实现这些门禁，当前配置不代表它们已经完成。

模型 x=[q,qdot,q_cmd-q]，每帧 21 维，窗口 [50,21]，标签末帧实测关节力矩 7 维。归一化只 fit train，左右独立 run；stateless 单向 LSTM 128/2，head 256/2，dropout 0.1。输出 root `/next/{side}`，三路力矩值按 NEXT 上游兼容约定放 JointState.position，单位 Nm，name 需为正确 7 关节（由第 6 部分实现）。残差是 measured - free_hat，为关节力矩估计。

硬件健康 `/factr2/w3_health` 为 DiagnosticArray，status `factr2/w3/left` 和 `factr2/w3/right`。header 是诊断生成 ROS 时间；raw_source_stamp_ns 是 bridge 发布 ROS 时间，raw_receive_age_seconds 是 monotonic 接收年龄，并需检查 ROS 源年龄。每侧 channel=0/1、slot=0..6 齐全唯一、有限、online、enabled、ERR=1 且原始输入未过期才 OK。新诊断 header 不得刷新缓存 raw 的有效期；缺少健康不是健康。真实 profile 必须 require_hardware_health=true，mock 可 false，默认 health timeout=0.25 s。诊断不证明每个电机真实接收时间，也不能发现所有 online=true 的数值冻结。

adapter_status 位于 `/factr2/{side}/adapter_status`，header 为生成时间，包含两源 age、skew、有效输出频率、累计 invalid/stale/duplicate/skew/health_rejected 计数和 reason。参数在启动时固定，动态修改拒绝。

路径：`data/{side}/{session_id}/`、`runs/next_{side}_{dataset_id}_{timestamp}/`、`reports/generated/`，均忽略 Git。NEXT 普通 YAML 经 config_file 参数传入，trainer 经 --config；不能将普通 YAML 当 ROS --params-file。adapter 的配置为 ROS 参数 YAML。

后续 manifest 必须含 contract_version、side、joint_names、sample_hz、history、session_id、dataset_id、episode/split 清单、自由空间/接触标记、工具/负载、两仓库 commit、标定文件路径+SHA256、实际数据绝对路径+SHA256、起止时间、质量统计、采集配置。模型 metadata 必须含 contract_version、side、joint_names、input_size/output_size/history、feature_order、模型配置、train/val session IDs、manifest SHA256、归一化来源 train、软件/依赖版本、checkpoint/config/normalization/metrics 绝对路径+SHA256；工具在第 4/5 部分实现。

### 数据集与 checkpoint 实现（第 4/5 部分）

录制侧车为同名 `.metadata.json`，严格检查模块为 `factr2_next.data_collection.quality`。`w3_split_v1` manifest 保存 dataset_id、三份非空 train/val/test、每份绝对 path、内容 SHA256、episode 清单及 rosbag audit_paths。唯一来源是 `(H5 SHA256,episode)`。未通过严格检查/标记排除的片段不能用于训练。

`w3_checkpoint_v1` 的 `metadata.json` 记录 contract_version、side、joint_order、feature_order=[q,qdot,q_cmd-q]、history=50、input_size=21、output_size=7、sample_hz=50、dataset_id、manifest_sha256、tool/gripper/load/calibration、software_versions/commits、seed、resolved_device 和 artifact_sha256。加载时逐项核验 config/权重/归一化及运行 profile，不以尺寸相同代替侧别匹配。保存 minimum validation normalized MSE 的 best epoch，test 仅冻结后评估。

x normalization 的统计覆盖训练窗口每个时间步（按重叠次数加权），y normalization 覆盖训练窗口末帧标签；float64 计算、float32 存储、std+1e-6。离线残差 measured-predicted，单位 Nm；physical MSE 为 Nm²。部署与离线都使用 stateless 同一窗口，不保存跨窗口 recurrent hidden state。

### 在线有效性和状态（第 6 部分）

`/next/{side}/status` (`diagnostic_msgs/DiagnosticArray`) 的 name 为 `factr2/next/{side}`。每 20 ms 输出 JSON 编码的 state/reason/history_count/source_age_seconds/receive_age_seconds/last_output_stamp_ns/output_hz/infer_ms/infer_p95_ms/outputs/dropped/resets/device。状态 loading/warming/valid/stale/invalid；invalid/stale 不发布 torque，恢复从 50 帧新历史开始。header 为诊断生成时间，last_output_stamp_ns 才是最近有效 torque 的末帧源时间。

W3 推理必须使用对应 side 的 checkpoint 和新鲜 adapter OK diagnostics；四输入严格同 stamp、同有序 name，gap>40 ms/坏帧/回退/adapter 非 OK 重置 history、filter、contact。source+monotonic receive 超时=.25 s，steady watchdog 周期=.02 s。三路 torque name/order/末帧 stamp 完全相同，position 单位 Nm；无 header 标量保持上游兼容，通过诊断关联有效性。
