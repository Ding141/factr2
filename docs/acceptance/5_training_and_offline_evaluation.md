# 第 5 部分验收：训练、模型契约和离线评估

2026-10-05，FACTR2 `dingyj-dev`，前置提交 `f063b57`；W3 `d32262d`，本阶段无改动。**结论：PASS，范围：PASS_OFFLINE**，使用 CPU 合成数据。真实物理门槛保持 UNCONFIGURED，真机验收留待第 7/8 部分。

| 验收条目 | 命令/实验 | 证据 | 结论 |
|---|---|---|---|
| TRN-01/02 | `scripts/next.sh python -m pytest factr2_next/src/factr2_next/test/test_training_contract.py -q`：人工 70 行 ×2 episode、坏数据实验 | [5_evidence.json](5_evidence.json)：每段 21 个窗口、cmd-pos、末帧标签准确，第 22 窗口回到第二 episode；strict 拒绝 gap/异长，无截断 | PASS |
| TRN-03 | `scripts/next.sh python scripts/prepare_w3_smoke.py`；同一测试将 test torque 改为 +100 后重建 manifest、同 seed 再训练 | [5_summary.json](5_summary.json) 及 [5_evidence.json](5_evidence.json)：独立 recorder session；全部权重和 norm 逐值不变 | PASS |
| TRN-04 | `scripts/next.sh python -m factr2_next.training.train --config config/w3/left/smoke_train.yaml`，右侧配置同样执行 | [5_summary.json](5_summary.json)：CPU/单线程、seed=0、两 epoch、batch=64、默认 stateless LSTM，有限 loss、完整 artifact/metadata | PASS |
| TRN-05/06 | 同一 training_contract 测试：保存重载、dataset batched eval 与 HistoryBuffer 逐窗口比较 | [5_evidence.json](5_evidence.json)：allclose(1e-5)，末帧标签和 measured-predicted residual 一致 | PASS |
| TRN-07 | 同一测试的无前缀/显式 arm keys；`scripts/next.sh python scripts/evaluate_w3_smoke.py --side left`，右侧同样执行 | [5_summary.json](5_summary.json)：1105 行得到 1056 预测，60 s 录制左 2951/右 2950 预测，无隐式 1000 上限 | PASS |
| TRN-08 | 同一 training_contract 测试构造缺 artifact、错 side/order/rate/dim/std/权重 shape | [5_evidence.json](5_evidence.json)：加载前明确拒收，运行期 expected side 拒绝错臂 | PASS |
| TRN-09 | 同一测试手算物理指标；左右 `evaluate_w3_smoke.py` 生成报告、CSV、图表 | [5_evidence.json](5_evidence.json)：MSE/bias/std 一致，Nm 单位、逐样本 stamp、单 episode 频谱、分组和 train-mean baseline 完整 | PASS |
| TRN-10 | 同 seed CPU 重训实验；`scripts/next.sh python scripts/test_training_resources.py` | [5_summary.json](5_summary.json)：逐值复现；180000 行/179951 窗口，cache 20.16 MB 对照 eager 755.79 MB，加载+norm 0.123 s，峰值 329.4 MiB | PASS |

核心回归共 36 项通过（含第 4 部分 21 项）。额外构造 epoch 2 更差的测试，实际保存 epoch 1 权重，证明部署取 best 而非 last；真实两轮 run 均 best_epoch=2。`colcon build` 两包成功，左右配置模板检查通过。CLI 在 acceptance 模板 null 时产生 UNCONFIGURED JSON 并退出 2，不能自动 PASS。

| Synthetic smoke | 左 | 右 |
|---|---:|---:|
| train windows | 2951 | 2950 |
| val windows | 550 | 551 |
| test predictions | 551 | 551 |
| training seconds | 8.36 | 8.99 |
| peak RSS MiB | 405.4 | 407.5 |
| test model RMSE Nm | 0.000431 | 0.000559 |
| test mean baseline RMSE Nm | 0.021164 | 0.021392 |

这些数值来自可计算的 mock torque 公式，只说明流水线和度量实现正确，不表示真机性能合格。

可重载模型：

- `runs/next_left_synthetic_smoke_20261005_155856_184675`，model SHA256 `11b718ccfb3631e62852eec14ebcb64077b95d4cbe2259a9bddb274d91caa6e1`
- `runs/next_right_synthetic_smoke_20261005_155856_812052`，model SHA256 `ff7426690820d5a9917a2abfc51bd5912363cbefc3e41a6959ae69f62c7398c4`

复现和 CLI 兼容变化见 [训练/评估说明](../training_and_evaluation.md)。完整摘要 [5_summary.json](5_summary.json)，哈希证据 [5_evidence.json](5_evidence.json)。run/data/图表未提交二进制，留在项目内忽略目录；所有新增依赖使用已有隔离环境，Matplotlib cache 明确位于项目。
