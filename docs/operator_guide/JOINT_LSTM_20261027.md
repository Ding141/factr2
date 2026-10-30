# 七个独立 LSTM：离线训练版本

日期：2026-10-27。当前完成代码及离线验证，尚未执行完整实测数据训练、部署或机械臂运动。

## 论文对应关系

用户提供的 `ICRA_2020__Final_(1).pdf` 是 Yilmaz、Wu、Kazanzides、Tumerdem 的
“Neural Network based Inverse Dynamics Identification and External Force Estimation on the da Vinci Research Kit”。
第 3 页 III-A、图 2/4：六个网络分别预测六个关节，每个网络读取全部六关节的角度和速度。
原文使用 12→100→1 前馈网络、tansig 隐藏激活及 Bayesian Regularization，并非 LSTM。

本实现借鉴其独立单输出分工，保留 NEXT 的时序输入及优化器，不声称完全复现论文：

- 100 Hz、50 帧、21 维：七关节 q、qdot、q_cmd-q；标签不作为输入。
- 每个分支为独立的两层 128 单元单向 LSTM，输出头 128→256→1。
- 七个分支均读取全部 21 维，保留关节耦合信息。
- 不共享循环层、隐藏状态或输出头。每次调用重置循环状态。
- 输出按 right_joint_0…6 拼接为七维空载力矩；外力矩仍为反馈减去预测。
- 参数合计 1,698,823；旧共享模型为 244,231，约 6.96 倍。

## 独立训练

模型类型为 `joint_lstm`，W3 配置必须显式标记
`trial.architecture: independent_joint_lstm_v1`。原 `lstm` 配置和权重兼容路径保留。

每个关节都有独立 AdamW 优化器、物理 Nm² MSE、梯度裁剪、验证集最优轮次和早停。
一个关节的误差不会参与另一个关节的梯度裁剪。最佳组合允许七个分支来自不同轮次。
`metrics.json` 的 `per_joint_best_epoch` 和 `selected_validation_mse` 为最终组合的选择依据；
`best_epoch` 仅表示最后一次更新该组合的轮次，不能解释为全部关节共享最佳轮次。
某个分支早停后，恢复其最佳权重且不再更新；其后训练损失记录为 null。

均值及标准差只从训练集估计。数据校验会审计清单内所有文件，但训练及选择仅使用 train/val。
没有随机拆分重叠窗口，也不会重新划分已有采集记录。默认读取：
`data/right/wide_highspeed_mean_20261026_v1/split.json`。

配置为单随机种子 0、最多 60 轮、逐关节早停 patience=10、batch=256、CPU 8 线程。
本次从独立随机权重训练；没有将旧共享模型的权重伪装成七个已训练模型。
该配置使用完整训练窗口，并非上次 anchored 微调的子集/蒸馏目标；比较结果时需明确这一差异。

在 `/home/venom/factr2` 执行完整训练：

```bash
bash scripts/next.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m factr2_next.training.train \
  --config config/w3/right/train_joint_lstm.yaml
```

输出是 `runs/right_joint_lstm_v1_<时间戳>/` 下的一个联合检查点，内部包含七组独立参数。
无需启动七个 ROS 节点，也无需把同一份输入数据复制七次。
可在训练期间创建 `runs/STOP_JOINT_LSTM`，在当前轮完成后保存已选最佳组合并结束；
若启动前该文件已存在，训练会拒绝启动。重启不是断点续训，请先检查已有输出。

## 推理及部署边界

标准 `load_checkpoint` 和 `InferenceNode` 可加载这个七维均值模型，保留现有时间戳、
输入新鲜度、关节顺序及检查点哈希检查。完整训练保存阶段会执行真实加载器的回读一致性校验。

当前 frozen-sigma 研究窗口的方差头依赖旧共享 LSTM 的隐藏特征，不能直接读取新检查点。
旧 NEXT-U/frozen-sigma 入口继续明确拒绝不匹配架构。当前补偿默认配置和旧模型未切换。
部署新模型前应单独评估低速/高速误差、重新拟合匹配的方差头与残差补偿，或先用标准均值
推理路径评估未经补偿的残差。七个输出来自独立网络不等于预测误差在统计上彼此独立。

## 验证

测试覆盖：独立参数与输出顺序、全部关节输入可达性、无跨窗口状态、单关节标签扰动下
其他关节更新保持一致、逐关节最佳轮次和早停、完整合成数据训练与检查点回读、改变测试
标签不影响训练权重或归一化、架构身份校验，以及旧 NEXT-U/方差相关回归。

已校验真实数据清单及独立记录划分，并用训练记录窗口检查输入和输出维度/有限值。
此检查不代表已在真实数据上完成训练或验证精度。

CPU batch=1、50 帧微基准（预热 10 次、测量 100 次）：

| 模型 | 线程数 | 中位数 ms | P99 ms |
|---|---:|---:|---:|
| 共享 LSTM | 4 | 0.311 | 0.332 |
| 七个独立 LSTM | 1 | 2.867 | 2.935 |
| 七个独立 LSTM | 4 | 2.198 | 2.524 |
| 七个独立 LSTM | 8 | 1.995 | 3.468 |

这是空闲环境模型调用时间，未包含 ROS 调度、预处理、方差头和补偿，不能保证实机 100 Hz
端到端截止时间。部署时应重新测量，当前结果支持继续进行离线训练。

验证记录：`log/joint_lstm_implementation_20261027/checks.json`、`tests.log`。
