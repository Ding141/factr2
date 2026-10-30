# NEXT-U 受控消融（离线）

本实验只比较原 NEXT 的 MSE 训练与增加均值/方差输出后的 Gaussian NLL 训练。完整 MOB、MOB＋GRU 留给后续对照；现有 NEXT＋MD 中的 MD 是 Motion Discriminator，不是动量观测器。

已完成实现、自动化验证及真实数据的单 seed 预算缩减试验：两组均训练 33 轮，并完成离线评估。NEXT-U 测试 MSE 为 0.01553764 N·m²，NEXT_MSE 为 0.00680307 N·m²，本配置下未显示改进；未替换在线模型。原三 seed、120 轮上限的完整预算未继续执行，详情见下文预算缩减说明及实验目录的 `evaluation/INTERPRETATION_ZH.md`。

## 实验定义

|项目|NEXT_MSE|NEXT_U|
|---|---|---|
|输入|原 21 维 `[q,qdot,qcmd-q]`|相同|
|时序|100 Hz，50 帧，stateless LSTM|相同|
|均值路径|原 LSTM、原回归头|相同|
|输出|自由运动反馈力矩均值|自由运动反馈力矩均值与逐关节方差|
|训练目标|原实现的物理尺度 MSE，Nm²|逐关节 Gaussian NLL|
|方差头|存在但冻结，不解释或导出其方差|从原回归头最后隐藏层增加一个线性输出，参与训练|
|选模|验证集物理 MSE 最小，平手选最早|相同|

方差头不引入额外循环层、隐藏层或 dropout。本机 256 维回归隐藏层增加 `7×(256+1)=1799` 个可训练参数。两组的均值模型容量相同，总参数相同，但 NEXT-U 多出这些参与训练的方差参数；报告会列出总参数和可训练参数，不能宣称有效参数量完全相等。

这项消融回答的是“均值/方差联合学习及 NLL 重加权是否有用”，不能把变化只归因于方差头本身。后续如要拆解原因，可以增加固定方差 NLL、同方差可学习 NLL 等对照。本轮不引入它们或其他特征变化。

均值与外力矩保持原定义：

```text
mu_Nm = mu_z * y_std + y_mean
var_Nm2 = var_z * y_std**2
sigma_Nm = sqrt(var_Nm2)
external_hat_Nm = measured_Nm - mu_Nm
NLL = mean(0.5 * [log(2*pi*var_Nm2) + error_Nm**2/var_Nm2])
```

方差采用 `softplus(raw)+1e-4`，在标准化目标的平方单位下上限为 `1e4`，初始值为 1。下限和上限属于数值保护，不是传感器精度范围。可在输入配置的 `next_u: {floor, ceiling, initial}` 显式修改；准备后配置即冻结。多个实验若修改这些值，必须记录为不同实验。

训练和评估均使用原始关节力矩残差，不进行 EMA、自动调零、Jacobian 求逆或力矩阈值截断。尚未校准的 sigma 不能自动作为“95% 外力置信度”；本实验的高斯区间覆盖率只是诊断。较宽区间可能提高覆盖率，却没有改善均值精度，应结合区间宽度、NLL、RMSE 和偏差判断。

## 如何保证配对

- 重新训练两个实验组，不把以前不同代码版本训练的模型混作本轮配对基线。历史模型配置作为设置来源。
- 每个 seed 使用同一个初始化模型复制出两组；原 LSTM 与均值头逐权重一致。方差头初始化不消耗均值初始化或 dropout 的随机流。
- 两个 DataLoader 各自使用相同 seed 的独立 Generator，shuffle 顺序一致。每个 epoch 为两组设置相同 dropout 随机流；验证的 DataLoader 不消耗训练随机流。
- 训练集独自拟合归一化；两组及 val/test 使用相同数值，训练时重新拟合并与冻结值逐元素比对。
- 每个 seed 两组按 epoch 成对运行。训练轮数、batch、Adam/AdamW、学习率、权重衰减、梯度裁剪保持一致。只有两组均达到验证 MSE 的共同 patience 条件时才一起早停，故该 seed 两组接受相同轮数的训练。相较旧版各自早停，这是一条明确的新配对协议。
- 两组均按验证物理 MSE 保存最佳模型。NEXT-U 同时记录 NLL，但不用测试集或测试覆盖率挑 epoch、seed 或方差参数。测试评估单独显式运行。
- 沿用严格四路同时间戳 H5 校验，禁止插值、跨 episode 拼接或随机滑窗分割。按原始父会话 ID 和父 H5 哈希检查 train/val/test 隔离，并核对实际控制配置与标定一致。
- 冻结配置、划分、归一化、Python 代码及其 SHA256，同时锁定引用的 H5 和 sidecar。训练/评估自动在冻结代码中运行；数据/代码变化时报错。真实数据原文件不复制，必须保留原路径。
- 默认 seeds 为 `0 1 2`。单 seed 可做流程试验，不足以形成稳健结论。多个 seed 仍不等于多个独立机械臂会话，重叠窗口也不能当成独立重复实验。

本流程当前限定 CPU，检查 Python/NumPy/PyTorch/h5py 版本一致。固定线程数并启用 PyTorch deterministic 算法；这不承诺跨 CPU 型号/数值库版本逐位一致。真实数据原有会话已经被查看过，报告明确标为回顾性比较，不宣称新盲测。

## 文件与命令

源码：

- `factr2_next/src/factr2_next/factr2_next/training/next_u.py`：分布模型、原尺度损失、物理量恢复、指标和离线窗口预测。
- `scripts/run_w3_next_u_ablation.py`：prepare / verify / train / evaluate，以及独立实验 artifact 的存取。
- `scripts/test_w3_next_u_ablation.py`：配对一致性、梯度和单位、隔离与篡改检查、测试标签反事实检查、冻结代码端到端测试。

在 `/home/venom/factr2` 下准备实验：

```bash
scripts/next.sh python scripts/run_w3_next_u_ablation.py prepare \
  --config models/w3/right/targeted_candidate_20261016/config.yaml \
  --output data/right/next_u_ablation_20261023_v1 \
  --seeds 0 1 2
```

`prepare` 不开始训练。目标目录必须不存在，防止覆盖；当前已准备好的目录可直接验证：

```bash
scripts/next.sh python scripts/run_w3_next_u_ablation.py verify \
  --experiment data/right/next_u_ablation_20261023_v1
```

完整训练及评估为两个步骤：

```bash
scripts/next.sh python data/right/next_u_ablation_20261023_v1/start.py train
scripts/next.sh python data/right/next_u_ablation_20261023_v1/start.py evaluate
```

也可用源码入口的 `train --experiment DIR` / `evaluate --experiment DIR`；它们会自动转入冻结运行环境。默认每组最多 120 epochs、三个 seed，以冻结配置为准；未实测该实验总耗时。

如需停止，创建该实验根目录的 `STOP_TRAINING` 文件。当前成对 epoch 完成后停下，保留已完成的最佳快照，标记 `stopped`；原始流程不自动评估未完成实验。此版本不支持中断续训，不能删状态文件后拼接结果。预算缩减后的完整配对前缀可通过下述专用评估流程处理。

### 本次运行的预算缩减

用户在训练期间要求只保留一个 seed 并减少轮数。原冻结配置仍保留 `seeds=[0,1,2]`、每组最多 120 轮，未重写配置、数据锁或训练代码。实际 seed 0 在 33 轮触发两组共同验证 MSE 早停；停止请求抵达时 seed 1 已开始，其第一轮完整配对结束后停止。最终比较只使用 **seed 0 的 33 轮**；seed 1 的一轮记录保留但排除。

`budget_amendment.json` 记录请求、原数据锁哈希、实际完成轮数、排除的 seed 与预算评估代码快照校验和。目标原先拟定为 30 轮；实际前缀为 33 轮，此差异明确留档，不把已有权重改标为训练 30 轮。

`scripts/control_w3_next_u_budget.py` 可在首个模型完成目标 epoch 后发出停止文件，等待另一个模型完成同一 epoch；它不修改正在运行的优化器、shuffle 或 dropout 状态。评估时要求两组完整历史长度与 completion 一致，且快照确为该前缀上验证 MSE 最优的 epoch。默认评估入口仍拒绝普通 `stopped` 实验；专用预算入口必须提供明确的预算修订记录。

本次专用评估代码已保存在实验的 `budget_runtime/`，使用原冻结包，状态见 `budget_status.json`，日志见 `evaluation.log`。原 `state.json` 保持 `stopped`，评估完成状态另见 `evaluation_state.json`。结果是单 seed、训练期间缩减预算的回顾性试验，不能宣称多 seed 稳健性结论。

## 输出和判读

每个 seed 的 `results/seed_N/NEXT_MSE/` 和 `NEXT_U/` 保存：

- `model.pt`、`normalization.npz`、`config.yaml`、`metrics.json`、`metadata.json`：最佳 epoch 快照及校验和。使用独立 `next_u_ablation_v1` schema，标记 `deployment_ready=false`，现有 W3 在线加载器拒绝加载。
- 同目录上层的 `NEXT_MSE_history.json` / `NEXT_U_history.json`：完整逐 epoch 记录。最佳快照的 metrics 只保存截至最佳 epoch 的历史。
- `completion.json`：两组共同完成的轮数、停止原因。

评估结果在 `evaluation/`：

- `COMPARISON.md`：每个 seed 的测试 MSE 对比。
- `comparison.json`：逐关节 bias、MAE、RMSE、残差标准差、绝对误差 P95；按任务类型、速度标签、独立 H5 会话分组。NEXT-U 增加 NLL、平均 sigma、标准化残差及 68.27%/95%/99% 名义区间覆盖率与平均宽度。
- `seed_N_{val,test}_{NEXT_MSE,NEXT_U}.npz`：同一最终帧的自由力矩均值与原始外力矩残差。NEXT-U 另存物理方差 Nm²。两组保留相同的原始纳秒时间戳和 episode 索引，配套 `*_episodes.json` 指定 H5 校验和及 episode，禁止事后按不同“有效帧”子集评优。

报告的 `U minus MSE` 为负代表该测试集下 NEXT-U 的均值 MSE 较小。报告只对 seed 差异给描述统计，不用重叠窗口计算显著性，也不自动推荐部署。

先查看每个关节和每个独立会话，避免平均值掩盖个别关节退化。sigma 增大不能被当作外力为零，也不能只用 `abs(residual)/sigma` 作为报警。这里未加入独立标定集，也未对区间作缩放；后续需要新的独立无接触会话验证覆盖率，并用已知接触参考检验增益、持续加载、延迟、卸载恢复与误报。自由空间结果不能证明接触时的置信度，也不能证明电流反馈的准确性。

本轮使用旧数据包络，不能解决当前 J2/J4 在训练范围外的泛化问题。新增方差输出也不是 OOD 检测器，不能据此解除已有模型使用范围限制。

## 验证

```bash
scripts/next.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q scripts/test_w3_next_u_ablation.py
```

测试以临时目录中的合成数据完成短训练，包含冻结源码运行的两 seed 端到端检查；不会读取机械臂或下发任何控制命令。
