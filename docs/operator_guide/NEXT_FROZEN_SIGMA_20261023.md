# 冻结 NEXT 均值的方差消融与 J1/J4 诊断

本轮继承上一轮配对试验 `data/right/next_u_ablation_20261023_v1`，固定其中 seed 0 的 NEXT_MSE 验证最佳模型（epoch 13）。这指上一轮消融的基线权重，不代表当前在线部署权重。目标是拆开“力矩均值是否准确”与“是否能合理描述残差离散程度”，不实现完整 MOB＋GRU，也不修改在线模型。

本轮已完成：方差头 9 轮早停，epoch 4 验证 NLL 最佳，完整运行约 20.6 秒。两组与原 NEXT_MSE 回放均值差异为 0，测试 MSE 保持 0.00680307 Nm²。固定 sigma 的测试 NLL 为 −0.472874，学习 sigma 为 0.148548（越低越好）；当前学习 sigma 未显示总体优势。J1/J4 名义 95% 覆盖率分别由固定 sigma 的 57.80%/75.90% 变为学习 sigma 的 57.59%/71.92%。结果与判读见实验目录 `evaluation/COMPARISON_ZH.md`、`INTERPRETATION_ZH.md`。

## 两个对照

|项目|FIXED_SIGMA|LEARNED_SIGMA|
|---|---|---|
|均值|冻结的 NEXT_MSE 最佳模型|完全相同|
|sigma|每关节一个常数|原回归头最后隐藏表示输入线性方差头|
|拟合数据|仅训练残差|仅训练隐藏表示与残差|
|可训练参数|0|1799|
|选模|无需选 epoch|验证集物理尺度逐关节 Gaussian NLL|

固定方差取零均值高斯残差模型的最大似然解：`var_j = mean_train(error_j**2)`，包含均值偏差，不把残差先减均值后只计算标准差。它描述固定均值估计器的总残差，不能当作电机纯随机噪声。

状态相关方差输出为 `softplus(linear(hidden_standardized))+floor`，标准化目标的方差单位下 `floor=1e-4`、`ceiling=1e4`。线性头零权重初始化，偏置使初始输出等于固定方差对照。头输入的均值与标准差仅由训练隐藏表示拟合。恢复物理方差时乘原目标归一化 `y_std**2`。

默认 recipe 在查看本轮诊断前固定：seed 0、最多 20 epochs、batch 2048、AdamW、lr 0.001、weight decay 1e-6、梯度裁剪 1、验证 NLL patience 5、warmup 5、min_delta 1e-5。固定方差作为 epoch 0 候选；按验证 NLL 选最小值，平手保留较早 epoch。不会用测试覆盖率选择轮数或超参数。

冻结均值模型始终处于 eval/inference 模式，关闭所有 dropout。train/val 各运行一次 LSTM，缓存末层隐藏表示、均值和残差；优化器只接收新方差头参数。因此训练阶段不需要反向传播 LSTM。测试缓存只在方差头选模结束后生成。训练前后检查均值权重摘要一致，两组输出同一份均值和残差数组；其 MSE 相同是实验定义所保证的结果。

## 旧联合 NEXT-U 诊断

`scripts/diagnose_w3_next_u.py` 从上一轮逐帧数组和原 H5 提取精确对应的末帧。要求原始纳秒时间戳、episode 索引和反馈力矩一致，不做插值或跨 episode 延续状态。

重点分析 J1、J4，按以下条件统计 NEXT 与旧联合 NEXT-U 的 bias、RMSE、误差 P95，以及旧 sigma 的区间覆盖率和宽度：

- 静置、单关节、联动，及每个 episode。
- 由实测速度定义的正向、负向、近零速度，死区 0.1°/s；绝对实测速度分档。
- 由训练集关节位置分位数定义的姿态分箱。
- 超过训练集绝对跟踪误差分位数的片段。
- 静置 episode 最后最多 1 秒；它只是尾段标签，不保证速度严格为零。
- 实测速度越过死区的符号换向后 0.5 秒，episode 边界重置。

诊断还记录旧方差触及上下限的比例、最终帧状态超训练各轴 min/max 的比例，以及均值位移对 MSE 差异的代数分解。单轴包络不构成 OOD 认证，分组相关性也不能证明摩擦或损失权重是退化原因。

## 冻结与复现

新增代码：

- `factr2_next/src/factr2_next/factr2_next/training/frozen_variance.py`：均值缓存、固定方差、方差头与 NLL 训练。
- `scripts/run_w3_frozen_sigma.py`：准备、完整运行、artifact 校验与报告。
- `scripts/diagnose_w3_next_u.py`：旧试验的 J1/J4 分组诊断及图。
- `scripts/test_w3_frozen_sigma.py`：均值冻结、单位、已知异方差合成数据、逐帧对齐及冻结源码端到端检查。
- `scripts/summarize_w3_frozen_sigma.py`：只读后处理，统计 sigma 与绝对误差的关联、静置速度离散档位和位置跨度；不重新训练或选模。

在 `/home/venom/factr2` 下：

```bash
scripts/next.sh python scripts/run_w3_frozen_sigma.py prepare \
  --parent data/right/next_u_ablation_20261023_v1 \
  --output data/right/next_u_frozen_sigma_20261023_v1

scripts/next.sh python data/right/next_u_frozen_sigma_20261023_v1/start.py
```

准备后的目录必须保持完整，不覆盖已有结果。`start.py` 自动使用冻结源码；父试验的数据锁、权重、旧逐帧结果、依赖版本以及本轮代码与 recipe 都有校验和。重复运行会被拒绝。

## 输出与解释

- `state.json`、`training.log`、`progress.json`：当前阶段和训练进度。
- `mean_checkpoint/`：继承的均值模型副本及原 artifact 校验和。
- `cache/` 与 `cache_sha256.json`：一次性推理的隐藏表示、均值和残差缓存。
- `variance_head/`：方差头最佳权重、训练隐藏归一化、固定方差对照、完整 epoch 记录。方差头权重绑定原均值模型文件的 SHA256。
- `evaluation/COMPARISON_ZH.md`、`comparison.json`：覆盖率、宽度、NLL、均值 MSE，以及按任务类型、到位方向和 episode 的分组指标。
- `evaluation/previous_diagnosis.json`、`previous_test_diagnosis.png`：旧联合 NEXT-U 的 J1/J4 诊断。
- `evaluation/coverage_width.png`：固定与学习 sigma 的覆盖率、宽度对照。
- `evaluation/{val,test}_{FIXED_SIGMA,LEARNED_SIGMA}.npz`：两组相同均值、残差、时间戳和 episode 索引，以及各自物理方差 Nm²。
- `evaluation/INTERPRETATION_ZH.md`、`postcheck.json`：结论与信号核查。测试静置片段 J1/J4 速度中位数约 0.630°/s、0.280°/s，与 12-bit 对称速度编码的近零档位相符；尾段位置跨度分别至多约 0.02186°、0°（受位置量化限制）。因此预设 0.1°/s 死区的原始方向/换向分组只能视为反馈符号标签，不能证明真实运动方向。该后处理没有修改任何模型输入。

这些是已经查看过的历史会话，结果属于单 seed 回顾性试验。均值和方差的选模复用了同一验证集；训练集拟合的 sigma 也可能因训练残差较小而偏乐观，验证 NLL 更好不保证新会话覆盖率更好。这里没有独立区间校准，不能把名义 95% 区间写成“已验证 95% 外力置信度”。还需要新的独立无接触会话及已知接触参考，验证覆盖率、加载增益、延迟和卸载恢复。

只学习 sigma 不能修复均值偏差、位姿覆盖不足或电流/力矩标定误差，也不能通过放大 sigma 将外力判为零。新 artifact 标记 `deployment_ready=false`，不替换在线检测窗口。

```bash
scripts/next.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q scripts/test_w3_frozen_sigma.py scripts/test_w3_next_u_ablation.py
```
