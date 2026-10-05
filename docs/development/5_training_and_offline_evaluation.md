# 5. 可复现训练、checkpoint 契约与离线评估

## 目标与依赖

在合成数据上跑通数据质量门禁 → 训练 → 保存 → 重载 → 独立离线评估，为真机训练提供可靠工具。依赖 [1](1_environment_and_contract.md)、[4](4_recording_and_data_quality.md)。本阶段证明实现正确，不证明 W3 实际模型质量；真实指标在第 7 部分验收。

## 必读代码与现状

前缀：`factr2_next/src/factr2_next/factr2_next/`。

- `training/dataset.py`：每 episode 单独组窗口，不跨 episode；当前最短流截断、无时间检查、一次性堆叠所有窗口。
- `training/train.py::make_datasets,split_indices,fit_normalization,save_run`：支持独立 val 文件；保留 random window split；仅 train fit normalization 已正确。metrics 只保存各 epoch 平均归一化 MSE；当前保存最后 epoch，不挑 best。
- `training/models.py`：stateless LSTM 已实现，stateful 会抛 NotImplementedError。不要改默认模型语义。
- `inference/checkpoint.py`：重建模型、读取 normalization，但无 joint_order/side/sample_hz/contract 校验。
- `inference/offline_eval.py`：`--arm` 必填且 read_episode 硬编码 `{arm}_joint_pos` 等键；与当前规划的单臂无前缀录制文件不兼容，默认最多评估前 1000 步。
- `inference/history_buffer.py`：特征公式一致，但无 shape/finite/clear 验证；本部分负责训练/离线共用特征契约，第 6 部分负责在线断流策略。

## 实施步骤与文件范围

1. 训练/评估加载第 4 部分的 accepted manifest 和 strict 数据检查。W3 profile 不允许随机窗口 split；正式实验必须显式独立 session 的 train/val/test。禁止最短流截断“修好”不合格数据；输入同一文件的不同不重叠 episode 可用于测试，但正式建议不同 session。
2. 数据错误在训练前拒绝：shape、有限值、时间、contract、side、joint_order、工具/标定配置不一致。没有 val windows、history 过长或 episode 过短时给出明确原因；窗口数量 N-H+1，标签索引末帧 H-1..N-1。
3. 对 feature construction、训练归一化和推理重载共用契约；可提取小型 helper，保持上游通用模型/非 W3 用途兼容。normalization x_mean/x_std 为 21 维，y_mean/y_std 为 7 维、std>0 且有限。
4. 维持默认架构；smoke 可以用 1~2 epoch、小 batch，在独立 smoke 配置中声明，正式配置保持可追溯。训练仅使用实际配置的 device；CPU 回退若发生必须记入 resolved config 和报告，不能把 CPU 运行标成 CUDA。
5. 保存 `model.pt,config.yaml,normalization.npz,metrics.json`，另加 `metadata.json`。metadata 包含 side/joint_order/feature_order/history/input_size=21/output_size=7/sample_hz=50/contract_version/dataset_id/分割清单哈希/工具与标定标识/软件版本/seed/实际 device。config 保存解析后的实际值和数据路径。
6. 记录按 epoch 的 train/val loss，以独立 validation 的 loss 最低选择最终 checkpoint，并记录 best_epoch / selection_rule。test 只用于冻结后的最终报告，不参与归一化、阈值选择或 checkpoint 选择。若选择继续保存 last，则另命名并明确最终部署指向 best。
7. checkpoint 加载验证文件齐全、metadata 与 config/权重形状一致、归一化合法、side/order/rate/history 与运行 profile 相符。拒绝加载左臂模型到右臂或旧配置，不凭输入维数相等认为兼容。读取自己生成的受信 checkpoint；具体 torch.load 行为按第 1 部分实测版本验证。
8. 改造 offline_eval，默认使用 run config 的 data.keys / contract，支持当前单臂无前缀键及原有显式双臂键。支持多文件、多 episode、全量评估（取消隐式前 1000 步截断）与 JSON/CSV 输出；输入参数与 stdout 兼容变化写文档。
9. 以物理 Nm 反归一化后输出每关节 RMSE/MSE、signed bias、残差 std、绝对残差 P95/P99、样本数/时长。预测残差符号明确：raw residual=measured-pred。按 session/轨迹/速度/姿态分组，产生静态误差曲线和残差频谱；频谱不跨缺口或 episode。
10. 实现可复现比较基线：训练集每关节平均力矩常数预测器；相同验证/测试集比较 overall 及每关节 RMSE。Pinocchio observer 只可作独立诊断，不替代标签。
11. 注意 dataset 当前窗口全量复制的内存成本；为预计数据规模给出内存估算/预算并做代表性加载。若需要优化为惰性组窗，必须证明特征、末帧标签、顺序和训练归一化结果等价，不能为省内存改变模型输入。

建议新增 `docs/training_and_evaluation.md`、`test/test_dataset_contract.py`、`test/test_training_checkpoint.py`、`test/test_offline_eval.py`；实际 CLI 与共享 metadata schema 同步更新第 1 部分契约。

## 验收标准

| 编号 | 验证 | 必须结果 |
|---|---|---|
| TRN-01 | 人工可计算的 7 关节 episode | x 第三段恰为 cmd-pos，窗口第 i 标签恰为 i+H-1 的 torque；H=50/末帧边界正确 |
| TRN-02 | 多 episode / 文件和时间缺口 | 不跨 episode 组窗，不跨>40 ms 缺口；坏 shape/异长不截断成合法数据 |
| TRN-03 | 独立 train/val/test | 来源无重叠；验证值明显偏移时 norm 仍只由 train 计算，修改 test 不改变训练/模型选择 |
| TRN-04 | stateless LSTM smoke | 有限 loss，shape [B,50,21]→[B,7]，生成全部 run artifact 和 metadata |
| TRN-05 | 保存/重载一致性 | 同一窗口 eval 模式保存前后反归一化预测 allclose(atol=1e-5,rtol=1e-5)，输出单位 Nm |
| TRN-06 | dataset vs HistoryBuffer | 同一序列的每个末帧预测与离线滚动预测在上述容差内一致，raw residual 计算正确 |
| TRN-07 | offline_eval 兼容 | 无前缀四键、显式 arm-prefixed 四键均可评估；全量预测数=sum(N-H+1)，不是默默只取前1000步 |
| TRN-08 | 不兼容 checkpoint | 缺文件、错 joint_order/side/维数/rate、非法 std 均在启动/加载时失败，有明确原因 |
| TRN-09 | metrics 和基线 | JSON/CSV 的每关节指标与小型手算样本一致，单位与符号正确，baseline 只 fit train |
| TRN-10 | 复现与资源 | 同 seed/配置/CPU 数据训练可在合理数值容差内复现；记录时间/内存与真实 device |

- [ ] 核心训练/评估可离线运行，无 ROS graph、CAN、W3 启动需求；所有新增行为测试通过。
- [ ] 最终选择的 checkpoint 确为声明的 best epoch；metrics 有 best_epoch 和选择依据。
- [ ] 只将 synthetic run 标为 smoke，不报告“真机模型已合格”。

命令参考：

```bash
ros2 run factr2_next next_train --config /home/dingyj/factr2/config/w3/left/train.yaml
python -m factr2_next.inference.offline_eval --help
```

具体评估参数由本阶段实现后写报告，不能将现有 `--arm left` 命令直接用于无前缀单臂 H5。

## 真机模型门槛的工具支持

交付 `config/w3/acceptance.template.yaml`，包含 per_joint_rmse_nm、per_joint_abs_bias_nm、baseline_improvement、近常量关节 sigma_floor、无接触 contact false-positive time rate 等门槛字段。默认未测量的物理阈值必须为空并拒绝自动 PASS；第 7 部分根据实际噪声/工作需求冻结成有版本的验收配置。不能在看完最终 test 后放宽门槛。

## 交付与交接

提交训练/评估修正、metadata 契约、测试、smoke 配置、验收模板和 `docs/acceptance/5_training_and_offline_evaluation.md`。生成 run 放在 runs/，报告给出路径/哈希。第 6 部分得到可重载的 7 关节 smoke checkpoint；第 7 部分得到真实数据训练和评估命令。
