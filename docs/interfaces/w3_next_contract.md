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
