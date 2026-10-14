# FACTR2 NEXT + W3 Integration

这个仓库保存 FACTR2 NEXT 模型、数据处理、训练/推理配置和复现文档。

针对空载偏差、卸载残留和末端奇异提示，当前重采入口为 [直接末端位姿采集与训练](docs/operator_guide/NEXT_POSE_CAPTURE_20261014.md)：一次发送完整目标，训练检查实测速度覆盖和返回基线。10 月 14 日已在慢/快 10/12°/s 下完成 386 个真机动作；10 月 15 日完成单会话姿态留出的初步训练。该模型尚未通过独立会话、高速覆盖和接触卸载验收，不能据此判断原问题已解决。详细结果见[采集与训练记录](docs/acceptance/pose_trial_20261015.json)。

本项目必须与配套的 W3 控制仓库一起使用：

- FACTR2：`Ding141/factr2`
- W3 控制：`Ding141/dual_arm_robot`

本地推荐目录：

```text
/home/dingyj/factr2
/home/dingyj/w3_dual_arm_ws
```

W3 仓库负责 CAN、ros2_control 和真机状态；本仓库负责 NEXT 的数据采集、训练、推理以及后续 W3 数据适配。两个仓库通过 ROS 2 话题通信，不合并控制代码。

训练数据、虚拟环境和运行日志不提交到 Git；默认也忽略模型权重。10 月 10–12 日旧数据和权重已按用户要求清理，旧模型目录只保留报告与配置；[清理清单](docs/acceptance/cleanup_20261014.json)记录文件与占用。10 月 13–14 日最新数据和候选保留。复现步骤见 [FACTR2_NEXT_W3_REPRODUCTION_PLAN.md](FACTR2_NEXT_W3_REPRODUCTION_PLAN.md)。

供不同 agent 逐步执行的开发任务、前置依赖和验收标准见 [分步开发任务](docs/development/README.md)，共 8 个部分，建议按编号推进。各阶段区分无硬件开发验收与真机实验验收，具体报告由执行该阶段的 agent 填写。

第 1–6 部分已完成离线开发验收：

- [环境与统一契约](docs/acceptance/1_environment_and_contract.md)
- [W3 控制目标只读接口](docs/acceptance/2_w3_command_state.md)
- [状态适配与反馈健康检查](docs/acceptance/3_adapter_and_health.md)
- [录制与严格数据门禁](docs/acceptance/4_recording_and_data_quality.md)
- [训练、模型契约与离线评估](docs/acceptance/5_training_and_offline_evaluation.md)
- [只读推理与七关节页面](docs/acceptance/6_inference_and_visualization.md)

运行环境见 [隔离环境说明](docs/environment.md)，adapter/健康器入口见 [adapter README](factr2_w3_adapter/README.md)。真实采集须使用 real profile 并启用健康门禁；真机验证留第 7/8 部分。

现场人员从 [项目讲解与操作指导](docs/operator_guide/README.md) 开始：包含已完成的工作、隔离环境、W3 网页操作，以及第 7 部分采集/训练和第 8 部分只读部署/接触验收的步骤。

当前本机右臂（ROS域74，100 Hz）从 [2026-10-10 连接与模型启动手册](docs/operator_guide/RIGHT_ARM_STARTUP_20261010.md) 开始；[TXT版](docs/operator_guide/RIGHT_ARM_STARTUP_20261010.txt)。
