# J1–J7 全关节残差补偿观察（2026-10-26）

本版本按用户明确要求启用右臂七个关节的检测残差补偿。它只修改外力矩估计，不给电机增加补偿命令，也不替换控制器中的重力补偿。

- 冻结 NEXT 模型：`models/w3/right/wide_highspeed_observer_20261026_v1`。
- 七轴补偿文件：`data/right/wide_highspeed_compensation_all_joints_20261026_v1`。
- 原验证选择版本仍保留：`data/right/wide_highspeed_compensation_20261026_v1`，只启用 J1/J3/J5。
- 新版本只改变 `enabled_joints` 掩码；原有权重、支持域、限幅、12Hz 输入滤波和5Hz输出滤波完全保留。原验证结果及用户指定覆盖原因保存在 `activation_override`，不是重新通过验证选择。
- 检测窗口：<http://127.0.0.1:8083/>；原始NEXT对照窗口：<http://127.0.0.1:8082/>。
- 此次实机运动和完整波形：`log/wide_all_joint_compensated_observation_20261026_v2`。

已有保留测试集的结果：同滤波高速空载整体 RMS 约0.51941→0.39094Nm，低速约0.07725→0.09630Nm。低速回归未通过，`deployment_ready=false`；未验证接触力矩的准确性，原sigma不能当作补偿后置信区间。不在线学习、不自动归零；越出适用范围时保留回退与状态标记。本次实机记录不回灌拟合。

运行检测服务的入口为 `scripts/observe_w3_kinematic_compensation.py`，使用新 `--model-label` 参数可明确标识全轴研究对照。必须先确认旧8083服务已经正常退出，不能启动多个同话题发布者。已执行的 `run.py` 拒绝重复执行；再次运动需要建立新记录并完成现场和运行状态核查，不能删除运行状态文件绕过限制。

首次尝试在preflight阶段因左臂反馈新鲜度保护中止，未执行轨迹，记录保留在v1目录。`CoverageClient.ready()` 原来仅等待右臂，现同时等待碰撞所依赖左臂q/cmd/health新鲜且健康。250ms新鲜度、左臂静止姿态、原有碰撞/跟踪/CAN/守护保护均保持。新增缺失/过期/故障反馈回归测试，与原目标轨迹采集测试合计21项通过。附加只读服务超时的原始记录也已保留。

运动期间自动校时由现场人员暂停，完成并确认机械臂静止后恢复：`sudo systemctl start systemd-timesyncd`。不要在运动时恢复，以免时钟阶跃破坏绝对时间轨迹。
