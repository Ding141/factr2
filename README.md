# FACTR2 NEXT + W3 Integration

这个仓库保存 FACTR2 NEXT 模型、数据处理、训练/推理配置和复现文档。

本项目必须与配套的 W3 控制仓库一起使用：

- FACTR2：`Ding141/factr2`
- W3 控制：`Ding141/dual_arm_robot`

本地推荐目录：

```text
/home/dingyj/factr2
/home/dingyj/w3_dual_arm_ws
```

W3 仓库负责 CAN、ros2_control 和真机状态；本仓库负责 NEXT 的数据采集、训练、推理以及后续 W3 数据适配。两个仓库通过 ROS 2 话题通信，不合并控制代码。

训练数据、模型权重、虚拟环境和运行日志不提交到 Git；详见 `.gitignore`。复现步骤见 [FACTR2_NEXT_W3_REPRODUCTION_PLAN.md](FACTR2_NEXT_W3_REPRODUCTION_PLAN.md)。

供不同 agent 逐步执行的开发任务、前置依赖和验收标准见 [分步开发任务](docs/development/README.md)，共 8 个部分，建议按编号推进。各阶段区分无硬件开发验收与真机实验验收，具体报告由执行该阶段的 agent 填写。

第 1–3 部分已完成离线开发验收：

- [环境与统一契约](docs/acceptance/1_environment_and_contract.md)
- [W3 控制目标只读接口](docs/acceptance/2_w3_command_state.md)
- [状态适配与反馈健康检查](docs/acceptance/3_adapter_and_health.md)

运行环境见 [隔离环境说明](docs/environment.md)，adapter/健康器入口见 [adapter README](factr2_w3_adapter/README.md)。真实采集须使用 real profile 并启用健康门禁；真机验证留第 7/8 部分。
