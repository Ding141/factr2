# 第 5 部分验收：训练、模型契约和离线评估

2026-10-05，FACTR2 `dingyj-dev`，前置提交 `f063b57`。无硬件、CPU 合成数据开发验收通过。真实物理门槛保持 UNCONFIGURED。

| 验收项 | 结果 |
|---|---|
| TRN-01/02 | 人工 70 行 ×2 episode：每段 21 个窗口、特征 cmd-pos、末帧标签准确；第 22 个窗口回到第二 episode，第 1 段历史不会混入。共享 strict 拒绝 gap/异长等坏数据，禁止截断 |
| TRN-03 | train/val/test 来自第 4 部分实际 recorder 的独立 session。单测将 test torque 改为 +100 并重建 manifest，相同 seed 的全部权重和 norm 逐值相同 |
| TRN-04 | 左右 CPU/单线程、seed=0、两 epoch、batch=64、契约默认 stateless LSTM；有限 train/val loss，完整 artifact 和 metadata |
| TRN-05/06 | 保存前后模型 allclose(1e-5)；同序列 dataset batched eval vs HistoryBuffer 逐窗口 eval allclose(1e-5)，标签与 raw residual 符号一致 |
| TRN-07 | 无前缀和显式 `{arm}_...` keys；1105 行均得到 1056 预测。真实 60 s 录制全量评估左 2951、右 2950 预测，超过原 1000 限制 |
| TRN-08 | 缺 artifact、错 side/order/rate/dim、非法/错形 std、权重 shape 在加载前拒收；运行期 side expected 也拒绝错臂 |
| TRN-09 | 手算 MSE、bias、std 一致；Nm JSON/CSV、逐样本 stamp、单 episode 误差曲线/频谱、分组和 train-mean baseline 已生成 |
| TRN-10 | 同 seed CPU 逐值复现；180000 行代表加载严格 PASS，179951 窗口，cache 20.16 MB，对照 eager 755.79 MB，加载+norm 0.123 s，进程峰值 329.4 MiB（含 torch/夹具/检查） |

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
