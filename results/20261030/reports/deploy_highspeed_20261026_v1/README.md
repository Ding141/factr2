# 新模型只读部署（2026-10-26）

按用户明确要求“部署新的模型”，已启动新均值及配套方差头、速度/加速度补偿窗口。完整流程低速回归检查失败的原始结果仍保留，未修改阈值或标记为通过。窗口显示高速残差 RMS 下降 23.8%、旧低速 RMS 上升 9.9% 的已知限制。

窗口：http://127.0.0.1:8083/ 。J1/J3/J5 有资格启用新增补偿，其余关节仅使用新均值与相同滤波。所有新增补偿仍受训练姿态及运动特征范围限制。

部署后已检查：各数据源与模型发布器均只有一个，实际输出绑定新均值 metadata SHA256 077190d99440a8bf27dc7a41e03563528f1d630f2f56a37db87d3e1f5cc3ebdb，输入新鲜度约 6 ms。当前 J1/J2/J4/J5 超出训练姿态范围，因此状态为 out_of_scope，新增补偿回退、窗口显示不可用；没有移动机械臂或放宽范围检查。

本次没有修改控制器增益、模式和电机力矩命令。旧观察模型 models/w3/right/frozen_sigma_observer_20261023 与旧补偿 data/right/kinematic_compensation_20261025_v1 均保留以便回退。

运行进程和核查记录见 base_service.json、compensated_service.json、live_verification.json、deployment_status.json。deployment_ready=false 仍表示尚未满足完整性能和真实接触验证要求，不表示观察进程未启动。
