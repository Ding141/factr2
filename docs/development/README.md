# W3 / FACTR2 NEXT 分步开发任务

本目录依据 [整体规划](../../FACTR2_NEXT_W3_REPRODUCTION_PLAN.md) 和两个本地仓库的实际源码拆分。阅读基线：2026-10-05，FACTR2 commit `8434e19`，W3 commit `00740ac`。后续 agent 必须先检查当前代码和工作区改动，不能假定基线仍然有效。

这些文件是任务书，**不是已实现功能或已通过的验收报告**。实际实现状态以 [第1部分](../acceptance/1_environment_and_contract.md)、[第2部分](../acceptance/2_w3_command_state.md)、[第3部分](../acceptance/3_adapter_and_health.md)、[第4部分](../acceptance/4_recording_and_data_quality.md)、[第5部分](../acceptance/5_training_and_offline_evaluation.md)、[第6部分](../acceptance/6_inference_and_visualization.md) 等验收报告为准。

## 执行顺序与任务边界

| 编号 | 任务书 | 主要工作区 | 前置验收 | 硬件要求 |
|---|---|---|---|---|
| 1 | [环境与统一契约](1_environment_and_contract.md) | FACTR2 | 无 | 无 |
| 2 | [W3 控制目标只读接口](2_w3_command_state.md) | W3 | 1 的契约 | 无；真机观察留到 7 |
| 3 | [状态适配与反馈健康检查](3_adapter_and_health.md) | FACTR2 | 1、2 的接口；可先 mock | 无 |
| 4 | [录制链路与数据质量门禁](4_recording_and_data_quality.md) | FACTR2 | 1、3 | 无 |
| 5 | [训练与离线评估工具](5_training_and_offline_evaluation.md) | FACTR2 | 1、4 | 无；用合成数据 |
| 6 | [只读推理与七关节可视化](6_inference_and_visualization.md) | FACTR2 | 3、5 | 无；用回放和测试 checkpoint |
| 7 | [真机自由运动采集与模型验收](7_real_data_and_model_validation.md) | 两者分工 | 2、3、4、5 | 需要操作者和 W3 真机 |
| 8 | [真机只读部署与接触验收](8_readonly_deployment_and_contact_validation.md) | 两者分工 | 6、7；左右臂分别合格 | 需要操作者和 W3 真机 |

推荐按 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 逐步开发。若后续用户安排不同 agent 并行：1 后可并行开发 2 和 mock 驱动的 3，但 3 的最终验收依赖 2；5 后 6 和 7 可以独立推进，8 汇合。两个 agent 修改同一个共享文件前必须明确文件所有权。

整体规划 Part A/B/C/D/E/F 对应这里的 2 / 3 / 1+4 / 7 / 5+7 / 6+8。增加工具开发阶段是为了先消除上游与 W3 的实际差异，再做真机实验。

## 每个 agent 的工作方式

1. 阅读整体规划、本任务书和前置阶段的验收报告。检查两个仓库的 `git status`、相关源码和适用的 `AGENTS.md`。
2. 只做本阶段工作；跨阶段的接口调整要同步修改第 1 部分的契约和下游任务说明。保留他人未提交改动。
3. 为本阶段的关键行为提供自动化验证与可复现命令。mock、合成数据、真机结果分别标记，不用合成数据证明真实外力精度。
4. 将报告放入 `docs/acceptance/<编号>_<阶段>.md`，写明两个仓库的实际 commit/工作区差异、环境、命令、退出码、测试结果、证据路径、失败项和后续输入。
5. 报告结论只用 `PASS`、`FAIL`、`BLOCKED`。另按任务需要标明 `PASS_OFFLINE` / `PASS_HARDWARE` 范围；缺少依赖、数据或操作者的项目标记阻塞，不勾选通过。

每个报告必须有「验收条目 → 命令/实验 → 证据 → 结论」表。大体积数据、权重、venv、build/install/log 存在 FACTR2 的忽略目录，不提交 Git；提交配置、工具、清单和精简结果摘要。实际路径和 SHA256 写入清单，不能只写“已测试”。

## 共用约束

- W3 保持 Humble / 系统 Python 3.10；NEXT 独立 venv。两者通过 DDS 通信，构建和运行 shell 不混用 install overlay。
- NEXT 的 adapter、recorder、trainer、inference、visualizer 和健康检查器不拥有 W3 命令接口，不切控制模式、不调用使能/失能服务。
- W3 新增接口只观察 `JointPositionController` 已采用的目标，不改插值、增益、标定、CAN 或控制策略。
- 单臂固定 7 关节，左右分别训练、分别推理；`H=50`、50 Hz、stateless 单向 LSTM 为第一版基线。
- 自由空间训练标签是标定后实测关节力矩；外力估计是 `tau_measured - tau_free_hat`，不是六维末端 wrench。
- 真机运动与接触操作由现有 W3 流程和现场操作者执行。无硬件的开发 agent 必须先完成可交付的离线工具、配置和测试，再报告真机项待执行。

统一字段、时间和量化验收约定见第 1 部分。其余部分可以增加更严格标准，不能静默改变共享契约。

原规划未指定绝对力矩误差预算、接触成功率或完整时延门槛。本目录中的 40 ms 连续性限制、baseline 改善、时延和接触检出等数值是新增的工程验收起点，已在对应任务注明；物理误差预算必须用 pilot 和工作需求确定，并在最终测试前冻结。它们不是来自论文的已验证精度，也不是当前代码已经满足的性能。
