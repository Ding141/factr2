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
