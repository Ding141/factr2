# 2026-10-16 右臂待验收模型的真机观测

2026-10-23 窗口更新：下面的 8082 启动命令现在默认显示论文中的 J1–J7 外部关节力矩；以下末端六维显示描述保留为原实验记录。原始/滤波曲线、重启方式与可选末端模式见 [关节力矩窗口说明](JOINT_TORQUE_WINDOW_20261023.md)。

按用户要求，将本轮最佳第 12 轮模型用于真机只读观测。模型副本为 `models/w3/right/targeted_candidate_20261016`，来源和逐文件哈希见其中 `DEPLOYMENT_MANIFEST.json`。原始 run、模型元数据及 `deployment_ready=false` 保持原值；本次启动不表示完成正式验收。

关节窗口为 http://127.0.0.1:8081/，末端窗口为 http://127.0.0.1:8082/。末端窗口标注待验收模型，显示 Fx/Fy/Fz（N）和 Mx/My/Mz（N·m），坐标轴及作用点为 `right_attachment_point`。

本次部署配置：

- `config/w3/right/inference_targeted_20261016.yaml`
- `config/w3/right/endpoint_wrench_targeted_20261016.yaml`

部署前读取实际控制器参数、机器人描述和标定，按训练使用的 `arm_v2` 规则重算控制配置哈希，与模型记录一致：`cbfcb882c7148d63e304f96e920bc4f459dc74c978c32516d6f4a947fb20a1a5`。实际标定 SHA256 为 `edf771ba9b48ea3df508380569d22a30b0e51036dcb3a321df3cf9e582f2b9ca`。现场审计保存在 `log/deploy_targeted_20261016/runtime_audit.json` 及同目录参数快照；这些核对不能证明采样无时延。

运行于 ROS 域 74、localhost、100 Hz，使用 50 帧历史；保留 EMA alpha=0.2。运行范围直接复制模型中冻结的实际训练位置包络，余量为 0。末端使用 `full_wrench`，保留条件数 100、相对拟合残差 0.35 和阻尼 0.005。无效区间留空，不自动扣零。接触迟滞阈值沿用既有配置 1/2 Nm，尚非本模型独立标定的检出门限。

本次仅启动健康器、右臂适配器、推理/关节网页和末端网页，已有机器人位置保持及重力补偿继续运行。已启动进程及日志路径见 `log/deploy_targeted_20261016/processes.json`。重启前先核对现有进程，避免重复节点或端口；不要因停止观测而停止 W3 控制器。

已有健康器和 100 Hz 右臂适配器时，推理/网页可分别在终端启动：

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
scripts/next.sh ros2 launch factr2_next readonly.launch.py \
  side:=right \
  inference_config:=/home/venom/factr2/config/w3/right/inference_targeted_20261016.yaml \
  web_config:=/home/venom/factr2/config/w3/right/visualize.yaml
```

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
scripts/next.sh python scripts/w3_endpoint_wrench.py \
  --config config/w3/right/endpoint_wrench_targeted_20261016.yaml
```

关节推理有效不保证末端换算有效；末端还检查描述哈希、范围、同步、新鲜度、几何可观测性及末端载荷拟合。当前模型验证/测试仅覆盖 6°/s 参考姿态，尚需独立多速度、冷热机和接触—完全卸载验证。在线观测和窗口数值不能替代这些验收。
